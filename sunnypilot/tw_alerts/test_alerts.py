import ast
import hashlib
import json
import math
import unittest
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from .audio import GAP, GPS_SERVICES, ROOT, Chimes, gps_input
from .build_chimes import PATTERNS, chime
from .matcher import CAMERA_BEFORE, ENFORCEMENT, FAMILY, Matcher
from .build_points import camera_road
from .review_signs import corroborated
from .import_workbook import compile_rows
from .hud import Hud, KINDS, PACKET


def row(kind, distance, lateral=0, dirs=(0,)):
  return [24+distance/111320,121+lateral/(111320*math.cos(math.radians(24))),kind,list(dirs),0,'test']


def fix(m,t,y,x=0,speed=10,course=0,acc=4):
  return m.fix(t,24+y/111320,121+x/(111320*math.cos(math.radians(24))),speed,course,acc,5)


class Tracking(unittest.TestCase):
  def test_pass_boundary_and_no_replay(self):
    for kind in set(FAMILY)-{'sign'}:
      with self.subTest(kind=kind):
        m=Matcher([row(kind,50)])
        self.assertIsNone(fix(m,0,0))
        for t,y in enumerate([10,20,30,40,49,50,55,59.9],1):
          self.assertEqual(fix(m,t,y).kind,kind)
        self.assertIsNone(fix(m,9,60))
        self.assertIsNone(fix(m,10,65))

  def test_point_owns_sound_hud_distance_and_end_for_every_enforcement_type(self):
    for kind in ENFORCEMENT:
      with self.subTest(kind=kind):
        m=Matcher([row(kind,120),row('sign',0)])
        for t,y in enumerate(range(-200,311,5)):
          s=fix(m,t*.5,y)
          if -180 <= y < 130:
            self.assertEqual((s.kind,m.hud_kind,m.sound_family),(kind,kind,'fixed'))
            self.assertAlmostEqual(m.distance,120-y,places=5)
          else:
            self.assertIsNone(s)

  def test_400m_case_and_all_road_class_takeover_boundaries(self):
    for road,reach in CAMERA_BEFORE.items():
      with self.subTest(road=road):
        point=reach+100  # ordinary road: point is 400 m beyond sign
        m=Matcher([row('sign',0),row('mobile',point)+[road]])
        for t,y in enumerate(range(-30,int(point)+31,5)):
          fix(m,t*.5,y)
          tone='sign' if -10 <= y < 100 else 'fixed' if 100 <= y < point+10 else None
          self.assertEqual(m.sound_family,tone,(road,y))

  def test_new_sign_downstream_of_completed_camera_remains_available(self):
    for reverse in (False,True):
      direction=-1 if reverse else 1
      course=180 if reverse else 0
      for loc,expected in ((-3,None),(0,None),(3,'sign'),(18,'sign')):
        with self.subTest(reverse=reverse,loc=loc):
          m=Matcher([row('fixed',0,dirs=(course,)),row('sign',loc*direction,dirs=(course,))])
          for t,y in enumerate((-25,-15,-5,5,10,20)):
            fix(m,t,y*direction,course=course)
            if t:
              self.assertEqual(m.sound_family,'fixed' if y<10 else expected)

  def test_successive_points_continue_without_replaying_consumed_sign(self):
    m=Matcher([row('fixed',40),row('sign',0),row('mobile',85)])
    for t,y in enumerate(range(-30,301,5)):
      fix(m,t*.5,y)
      self.assertEqual(m.sound_family,'fixed' if t and y<95 else None)
      if y==50:self.assertEqual(m.hud_kind,'mobile')

  def test_losing_camera_without_passing_it_does_not_cancel_sign(self):
    m=Matcher([row('sign',0),row('mobile',160)])
    for t,y in enumerate((-25,-10,0,10)):
      fix(m,t,y)
    self.assertEqual(m.hud_kind,'mobile')
    for t,step in enumerate((1,2,3,4,5),4):
      fix(m,t,10+step*10,step*10,speed=15,course=45)
    self.assertEqual((m.sound_family,m.hud_kind),('sign','sign'))

  def test_sparse_fix_crosses_sign_and_camera_exit_together(self):
    m=Matcher([row('sign',0),row('mobile',3)])
    for t,y in enumerate((-50,-20,15,45)):
      fix(m,t,y,speed=30)
      self.assertEqual(m.sound_family,'fixed' if t==1 else None)

  def test_no_pair_for_signs_opposing_direction_and_side_road(self):
    for extra in [[],[row('sign',90)],[row('mobile',75,dirs=(180,))],
                  [row('fixed',75,dirs=(90,))],[row('red',75,lateral=60)],
                  [row('tech',75,lateral=20,dirs=())]]:
      m=Matcher([row('sign',0),*extra])
      for t,y in enumerate(range(-20,200,10)):
        fix(m,t,y)
        if y>=-10:
          self.assertEqual((m.sound_family,m.hud_kind),('sign','sign'))

  def test_unknown_direction_uses_narrow_corridor_and_cross_grid(self):
    for east in (True,False):
      sign=row('sign',0,dirs=());point=row('fixed',0,dirs=())
      sign[1]=121.0099 if east else 121.0101
      direction=1 if east else -1
      k=111320*math.cos(math.radians(sign[0]))
      point[1]=sign[1]+direction*60/k
      m=Matcher([sign,point])
      for t,d in enumerate((-25,-8)):
        m.fix(t,sign[0],sign[1]+direction*d/k,12,90 if east else 270,4,5)
      self.assertEqual((m.hud_kind,m.sound_family),('fixed','fixed'))

  def test_bend_hands_over_when_enforcement_itself_is_in_range(self):
    for distance,expected in [(145,'fixed'),(425,'sign')]:
      course=30; rad=math.radians(course)
      m=Matcher([row('sign',0),row('mobile',25+distance*math.cos(rad),distance*math.sin(rad),dirs=(course,))])
      for t,y in enumerate((-20,-10,0,10,25)):
        fix(m,t,y)
      self.assertEqual(m.sound_family,'sign')
      fix(m,5,25+10*math.cos(rad),10*math.sin(rad),course=course)
      self.assertEqual(m.sound_family,expected)

  def test_pair_state_not_inherited_on_reverse_or_gps_reset(self):
    m=Matcher([row('sign',0,dirs=()),row('mobile',100,dirs=(0,))])
    fix(m,0,-20);fix(m,1,-8)
    self.assertEqual(m.sound_family,'fixed')
    self.assertIsNone(fix(m,2,0,acc=55))
    for t,y in enumerate((22,8,-3),3):
      fix(m,t,y,course=180)
    self.assertEqual((m.sound_family,m.hud_kind),('sign','sign'))

  def test_separate_encounters(self):
    m=Matcher([row('mobile',35),row('sign',55),row('fixed',420)])
    kinds=[(s.kind if (s:=fix(m,i,i*10)) else None) for i in range(46)]
    self.assertEqual([v for i,v in enumerate(kinds) if i==0 or v!=kinds[i-1]],
                     [None,'mobile','sign','fixed',None])

  def test_sign_window_and_low_rate_crossing(self):
    for path in ((-22,-10.1,-10,0,20), (-42,-18,8,32)):
      m=Matcher([row('sign',0)])
      for t,y in enumerate(path):
        target=fix(m,t,y,speed=25)
        self.assertEqual(target is not None, y >= -10)
      t=len(path)
      for y in [*range(55,285,20),299.9,300,310]:
        target=fix(m,t,y,speed=25);t+=1
        self.assertEqual(target is not None, y < 300)
    for loc,heading in ((-100,0),(0,180)):
      m=Matcher([row('sign',loc,dirs=(heading,))])
      for t,y in enumerate((-20,-9,5,20)):
        self.assertIsNone(fix(m,t,y))

  def test_sign_follows_gentle_bend_and_pauses_when_stopped(self):
    m=Matcher([row('sign',0)])
    for t,y in enumerate((-20,-8,0)):
      fix(m,t,y)
    origin=m.distance; self.assertIsNotNone(fix(m,3,1,speed=0));self.assertEqual(m.distance,origin)
    x,y=0,1
    for n in range(1,16):
      course=min(70,n*5);step=20 if n<15 else 20.2
      x+=step*math.sin(math.radians(course));y+=step*math.cos(math.radians(course))
      s=fix(m,n+3,y,x,speed=20,course=course)
      self.assertEqual(s is not None,n<15)

  def test_camera_window_depends_on_road_not_car_speed(self):
    for kind in ('mobile','fixed'):
      for road,distance in CAMERA_BEFORE.items():
        for speed in (5,28):
          m=Matcher([row(kind,distance)+[road]])
          fix(m,0,-20,speed=speed)
          self.assertIsNone(fix(m,1,-.1,speed=speed))
          self.assertEqual(fix(m,2,0,speed=speed).kind,kind)

  def test_hud_handles_highway_distance_and_after_sign(self):
    hud=object.__new__(Hud)
    for kind,meters,label in [('fixed',int(CAMERA_BEFORE['freeway']),f"約 {int(CAMERA_BEFORE['freeway'])} m"),
                              ('sign',-100,'路段剩 200 m')]:
      hud.sock=Mock();hud.event=None
      hud.sock.recv.side_effect=[PACKET.pack(10.,KINDS.index(kind),0,meters),BlockingIOError()]
      self.assertEqual(hud.text(10.,10.,True)[1],label)

  def test_mainline_classification_excludes_ambiguity(self):
    for name,expected in [('國道一號北向83.4公里','freeway'),('國道三甲西向3.4公里','express'),
                          ('台74線35公里處','express'),('台61線70公里',None),('台74甲線3.4公里',None),
                          ('國道四號清水端與臨港路585巷口',None),('台61線29公里橋下',None),
                          ('中豐路與台68線路口',None),('中央路段速限90',None)]:
      self.assertEqual(camera_road(name),expected,name)

  def test_irrelevant_points_and_turnoff(self):
    for r in [row('sign',100,90),row('fixed',-20),row('mobile',50,dirs=(180,))]:
      m=Matcher([r]);fix(m,0,0);self.assertIsNone(fix(m,1,10))
    m=Matcher([row('fixed',80)]);fix(m,0,0);self.assertIsNotNone(fix(m,1,10))
    self.assertIsNone(fix(m,2,10,10,course=90))

  def test_stationary_fix_and_invalidity(self):
    m=Matcher([row('fixed',45)])
    self.assertIsNone(fix(m,0,0,speed=0))
    self.assertIsNone(fix(m,1,0,speed=0))
    self.assertIsNotNone(fix(m,2,8))
    self.assertIsNotNone(fix(m,3,8,speed=0,course=float('nan')))
    self.assertIsNone(fix(m,4,8,acc=25))
    self.assertIsNone(fix(m,5,10))
    self.assertIsNone(fix(m,8,20))
    self.assertIsNone(fix(m,9,400))


class SignReview(unittest.TestCase):
  def test_merged_workbook_policy(self):
    row={"新序號":1,"類型":"警52預告標誌","校正優先級":"OK_正常","heading_deg":0,"heading2_deg":180,
         "緯度":24.1,"經度":121.2,"速限":None,"資料集編號":"公路局警52","點位識別碼":"S1","地址或里程":"台1"}
    point,excluded=compile_rows([row])
    self.assertEqual((point[0][3],excluded),([0,180],[]))
    variants=[dict(row,新序號=2,校正優先級="P1_座標嚴重異常"),dict(row,新序號=3,校正優先級="P3_雙向軸向未知",heading_deg=None,heading2_deg=None)]
    p,e=compile_rows([row,*variants])
    self.assertEqual(([x[3] for x in p],len(e)),([[0,180],[]],1))
    for changes in ({"heading_deg":360},{"heading2_deg":35},{"緯度":0},{"heading_deg":None},{"校正優先級":"P2_無法推算方向"}):
      with self.subTest(changes=changes),self.assertRaises(ValueError):
        compile_rows([dict(row,**changes)])

  def test_independent_mileposts_and_direction(self):
    sign={'公路編碼':'X','起點樁號':'0K+500','坐標-N-WGS84':'24','坐標-E-WGS84':'121',
          '牌面方向':'順向','校正優先級':'OK_正常','heading_deg':'359'}
    sites={0:{(24-500/111320,121)},1000:{(24+500/111320,121)}}
    roads={'X':([0,1000],sites)}
    self.assertEqual(corroborated(sign,roads),359)
    for key,val in [('牌面方向','逆向'),('坐標-E-WGS84','120.99'),('heading_deg','38'),
                    ('校正優先級','P1_跳點'),('公路編碼','Y'),('起點樁號','9K+123')]:
      self.assertIsNone(corroborated({**sign,key:val},roads))
    self.assertEqual(corroborated({**sign,'牌面方向':'逆向','heading_deg':'180'},roads),180)
    sites[0].add((24,120.99))  # ambiguous/displaced reference on one side
    self.assertIsNone(corroborated(sign,roads))


class Audio(unittest.TestCase):
  def setUp(self):
    self.v=Chimes()
    self.v.matcher=Matcher([row('mobile',50)])
    class SM(dict):pass
    self.ext,self.internal=GPS_SERVICES
    g=NS(hasFix=True,latitude=24,longitude=121,speed=10,bearingDeg=0,bearingAccuracyDeg=5,horizontalAccuracy=4,source="ublox")
    modem=NS(**{**vars(g),"source":"qcomdiag","vNED":[10.,0.,0.],"horizontalAccuracy":0.,"bearingAccuracyDeg":180.})
    self.sm=SM({self.ext:g,self.internal:modem})
    self.sm.alive={self.ext:True,self.internal:False};self.sm.valid=dict.fromkeys(GPS_SERVICES,True)
    self.sm.updated=dict.fromkeys(GPS_SERVICES,True);self.sm.recv_time=dict.fromkeys(GPS_SERVICES,0.)

  def update(self,i,m,permitted=True,service=None):
    service=service or self.ext
    self.sm.recv_time[service]=i;self.sm[service].latitude=24+m/111320
    with patch(__package__+'.audio.notify'):
      self.v.update(self.sm,i,permitted)

  def test_loop_without_restarting_on_each_fix(self):
    self.update(0,0);self.update(1,10)
    key=self.v.request[1];sample=self.v.samples['fixed'];buf=np.zeros(len(sample)+GAP+len(sample))
    self.v.render(buf,True,1.01)
    np.testing.assert_array_equal(buf[:len(sample)],buf[-len(sample):])
    self.assertGreater(abs(buf[:len(sample)]).max(),.2)
    self.assertFalse(buf[len(sample):-len(sample)].any())
    self.update(2,20);self.assertEqual(self.v.request[1],key)
    for i,y in enumerate([30,40,50,59,60],3):self.update(i,y)
    self.assertIsNone(self.v.request)
    buf=np.ones(40);self.v.render(buf,True,7.1);self.assertTrue((buf==1).all())

  def test_disallow_and_stale_audio(self):
    self.update(0,0);self.update(1,10)
    for allowed,t in [(False,1.1),(True,3.)]:
      buf=np.full(20000,.3)
      self.v.render(buf,allowed,t)
      self.assertTrue((buf==.3).all())
    self.update(2,20,False)
    self.assertIsNone(self.v.request)
    self.update(3,30);self.update(4,40)
    self.sm[self.ext].hasFix=False;self.update(5,45)
    self.assertIsNone(self.v.request)

  def test_real_modem_defaults_to_zero_accuracy_and_heading_uncertainty_180(self):
    self.sm.alive={self.ext:False,self.internal:True}
    for t,meters in enumerate((0, 3, 8)):
      self.update(t,meters,service=self.internal)
      self.assertIsNone(self.v.request)
    self.update(3,15,service=self.internal)
    self.assertEqual(self.v.request[0].kind,'mobile')
    buf=np.zeros(4096);self.v.render(buf,True,3.05);self.assertGreater(abs(buf).max(),.2)
    for t,meters in enumerate((25,35,45,55,59.9),4):
      self.update(t,meters,service=self.internal)
      self.assertIsNotNone(self.v.request)
    self.update(9,60,service=self.internal)
    self.assertIsNone(self.v.request)

  def test_modem_cannot_play_without_good_motion_and_position(self):
    m=self.sm[self.internal];self.sm.alive={self.ext:False,self.internal:True}
    m.vNED=[-10.,0.,0.]  # reported south vs consecutive northbound fixes
    for t,y in enumerate((0,15,30,45)):
      self.update(t,y,service=self.internal);self.assertIsNone(self.v.request)
    m.vNED=[];self.assertIsNone(gps_input(m,self.internal))
    m.bearingDeg=-75;m.bearingAccuracyDeg=5
    self.assertEqual(gps_input(m,self.internal)[0][3],285)
    for name,bad in [('horizontalAccuracy',40),('horizontalAccuracy',float('nan')),('hasFix',False),('source','ublox'),('latitude',0)]:
      with self.subTest(name=name),patch.object(m,name,bad):
        self.assertIsNone(gps_input(m,self.internal))
    self.sm[self.ext].horizontalAccuracy=0;self.assertIsNone(gps_input(self.sm[self.ext],self.ext))

  def test_switch_gps_does_not_reuse_old_track(self):
    self.update(0,0);self.update(1,10)
    self.assertIsNotNone(self.v.request)
    self.sm.alive[self.internal]=True
    # External is stale; one modem fix must not inherit its alert.
    self.update(3,25,service=self.internal);self.assertIsNone(self.v.request)
    self.update(4,40,service=self.internal);self.assertIsNotNone(self.v.request)
    self.sm[self.internal].hasFix=False
    self.update(5,45,service=self.internal);self.assertIsNone(self.v.request)

  def test_only_two_tone_patterns_no_speech(self):
    self.assertEqual(set(PATTERNS),{'sign','fixed'})
    self.assertEqual({p.stem for p in (ROOT/'voice').glob('*.wav')},set(PATTERNS))
    for family,spec in PATTERNS.items():
      # Regenerate from pure notes and compare the entire shipped clip, not its prefix.
      pcm=chime(*spec)
      np.testing.assert_allclose(self.v.samples[family]*32768, pcm, rtol=0, atol=1)
      self.assertEqual(abs(pcm.astype(np.int32)).max(),21900)
      self.assertLess(abs(self.v.samples[family]).max()+.2,1.)  # full-scale optional overspeed mix
      self.assertLess(len(pcm),48000)

  def test_actual_mixer_keeps_native_alarm_and_quiet_mode(self):
    tree=ast.parse((ROOT.parents[1]/'selfdrive/ui/soundd.py').read_text())
    fn=next(i for i in ast.walk(tree) if isinstance(i,ast.FunctionDef) and i.name=='get_sound_data')
    ctx=dict(np=np,time=NS(monotonic_ns=lambda:0,monotonic=lambda:1.1),AudibleAlert=NS(none=0),
             AudibleAlertSP=NS(promptSingleLow=3),sound_list={2:('native',1,1)})
    exec(compile(ast.Module([fn],[]),'<soundd>','exec'),ctx)
    self.update(0,0);self.update(1,10)
    obj=NS(current_alert=2,should_play_sound=lambda a:bool(a),current_sound_frame=0,current_volume=1.,
           loaded_sounds={2:np.full(30000,.75),3:np.full(66,.25)},_sign_frame=0,_sign_until_ns=0,
           enabled=False,_tw_voice=self.v)
    for a,quiet,expected in [(2,False,.75),(0,True,0.)]:
      obj.current_alert=a;obj.enabled=quiet
      self.assertTrue((ctx['get_sound_data'](obj,3000)==expected).all())
    obj.enabled=False;self.assertGreater(abs(ctx['get_sound_data'](obj,4096)).max(),.05)

  def test_assets_and_snapshot(self):
    pts=json.loads((ROOT/'points.json').read_text());aud=json.loads((ROOT/'import-report.json').read_text())
    self.assertEqual(len(pts),5598);self.assertEqual(sum(x.get('points',0) for x in aud.values()),len(pts))
    self.assertEqual(sum(x.get('reviewed_reference',0) for x in aud.values()),276)
    self.assertEqual(len(json.loads((ROOT/'pending.json').read_text())),117)
    self.assertEqual(Counter(FAMILY[x[2]] for x in pts),{'sign':2568,'fixed':3030})
    signs=aud['169929-1.csv']
    self.assertEqual((signs['input'],signs['suspect_coord'],signs['duplicate'],signs['estimated_heading']),(2617,8,41,1469))
    self.assertFalse(any(r[:3]==[24.8786595,121.2093362,'sign'] for r in pts))
    review=json.loads((ROOT/'workbook-report.json').read_text())
    self.assertEqual(review['points_sha256'],hashlib.sha256((ROOT/'points.json').read_bytes()).hexdigest())
    self.assertEqual((review['input'],review['active'],len(review['quarantined'])),(5606,len(pts),8))
    self.assertEqual(Counter(x[2] for x in pts),review['by_kind'])
    self.assertEqual(Counter(str(len(x[3])) for x in pts),review['direction_counts'])
    self.assertEqual(sum(r[2]=='sign' and len(r[3])>0 for r in pts),2563)
    self.assertEqual(sum(bool(r[3]) for r in pts if r[2]!='sign'),2567)
    self.assertEqual(sum(x[2]=='mobile' for x in pts),277)
    self.assertEqual(set(self.v.samples),{'sign','fixed'})
    self.assertEqual(len(set(v.tobytes()[:8000] for v in self.v.samples.values())),2)

  def test_native_preemption_and_release_using_soundd(self):
    methods={'update_tw_chimes','get_sound_data','update_alert'}
    tree=ast.parse((ROOT.parents[1]/'selfdrive/ui/soundd.py').read_text())
    t=0.
    ctx=dict(np=np,time=NS(monotonic_ns=lambda:int(t*1e9),monotonic=lambda:t),
             AudibleAlert=NS(none=0,warningImmediate=9),AudibleAlertSP=NS(promptSingleLow=3),
             sound_list={2:('native',1,1)})
    exec(compile(ast.Module([f for f in ast.walk(tree) if isinstance(f,ast.FunctionDef) and f.name in methods],[]),
                 '<soundd>','exec'),ctx)
    frame=4096
    obj=NS(current_alert=0,should_play_sound=bool,current_sound_frame=0,current_volume=1.,
           loaded_sounds={2:np.full(frame,.73)},_sign_frame=0,_sign_until_ns=0,enabled=False,_tw_voice=self.v)
    state=NS(alive={'selfdriveState':True},valid={'selfdriveState':True})
    self.sm.update=lambda timeout:None
    def tick(t2,y):
      nonlocal t
      t=t2; self.sm[self.ext].latitude=24+y/111320; self.sm.recv_time[self.ext]=t
      with patch(__package__+'.audio.notify'):
        ctx['update_tw_chimes'](obj,state,self.sm)
    def render():return ctx['get_sound_data'](obj,frame)
    def native():ctx['update_alert'](obj,2); np.testing.assert_allclose(render(),.73)
    for kind,site in [('sign',20),('fixed',50)]:
      with self.subTest(kind=kind):
        self.v.revoke();self.v.matcher=Matcher([row(kind,site)])
        tick(0,0);tick(1,10);clip=self.v.request[2]
        np.testing.assert_array_equal(render(),clip[:frame])
        native();tick(2,20);tick(3,30)
        self.assertFalse(render().any())  # native one-shot finished, event not yet cleared
        ctx['update_alert'](obj,0)
        # No fresh fix since release: valid lease, sign is already behind the car.
        np.testing.assert_array_equal(render(),clip[:frame])
        native()
        for i,y in enumerate(range(40,site+(300 if kind=='sign' else 10)+11,10),4):tick(i,y)
        ctx['update_alert'](obj,0);self.assertFalse(render().any())
    for disabled,valid in ((True,True),(False,False)):
      self.v.revoke();self.v.matcher=Matcher([row('fixed',50)])
      obj.enabled=False;state.valid['selfdriveState']=True
      tick(0,0);tick(1,10);native()
      obj.enabled=disabled;state.valid['selfdriveState']=valid;tick(2,20)
      ctx['update_alert'](obj,0);self.assertIsNone(self.v.request);self.assertFalse(render().any())

  def test_soundd_feed_continues_point_wave_across_sign_until_point_exit(self):
    for kind in ('fixed','mobile'):
      self.v.revoke()
      self.v.matcher=Matcher([row(kind,90),row('sign',30)])
      self.update(0,0);self.update(1,10)
      gen=self.v.request[1]
      self.v.render(np.zeros(512),True,1.1)
      self.update(2,20)  # sign window opens, but HUD and audio stay with the point
      self.assertEqual((self.v.matcher.hud_kind,self.v.request[1]),(kind,gen))
      self.v.render(np.zeros(512),True,2.1);self.assertEqual(self.v.cursor,1024)
      self.assertIs(self.v.request[2],self.v.samples['fixed'])
      for t,y in enumerate(range(30,101,10),3):
        self.update(t,y)
      self.assertIsNone(self.v.request)  # no sign bell left for the same approach
    self.v.revoke()
    self.v.matcher=Matcher([row('sign',20)])
    self.update(0,0);self.update(1,10)
    self.assertIs(self.v.request[2],self.v.samples['sign'])

  def test_point_immediately_interrupts_fallback_audio_and_updates_hud(self):
    self.v.matcher=Matcher([row('sign',0),row('fixed',400)])
    for t,y in enumerate(range(-20,100,10)):
      self.update(t,y)
    gen=self.v.request[1]
    self.assertIs(self.v.request[2],self.v.samples['sign'])
    self.update(12,100)
    self.assertEqual(self.v.request[1],gen+1)
    self.assertEqual(self.v.matcher.hud_kind,'fixed')
    self.assertIs(self.v.request[2],self.v.samples['fixed'])
    out=np.zeros(2048);self.v.render(out,True,12.1)
    np.testing.assert_array_equal(out,self.v.samples['fixed'][:2048])


if __name__=='__main__':
  unittest.main()
