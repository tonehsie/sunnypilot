import ast
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

  def test_latched_sign_uses_shared_fixed_bell_for_every_enforcement_type(self):
    for kind in ENFORCEMENT:
      with self.subTest(kind=kind):
        m=Matcher([row(kind,60),row('sign',0)])
        for t,y in enumerate(range(-20,301,10)):
          s=fix(m,t,y)
          if -10 <= y < 300:
            self.assertEqual((s.kind,m.hud_kind,m.sound_family),('sign','sign_camera','fixed'))
          elif y==300:
            self.assertIsNone(s)

  def test_pair_is_relative_to_sign_not_vehicle(self):
    for pos,tone in [(-1,'sign'),(0,'fixed'),(299.9,'fixed'),(300,'fixed'),(300.1,'sign')]:
      with self.subTest(pos=pos):
        m=Matcher([row('sign',0),row('mobile',pos)])
        fix(m,0,-20);fix(m,1,-10)
        self.assertEqual(m.sound_family,tone)
        for t,y in enumerate(range(0,300,10),2):
          fix(m,t,y);self.assertEqual(m.sound_family,tone)

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
      self.assertEqual(m.hud_kind,'sign_camera')

  def test_bend_upgrades_only_when_remaining_zone_reaches_camera(self):
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
                              ('sign',-100,'路段剩 200 m'),('sign_camera',-100,'路段剩 200 m')]:
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
    self.assertEqual(len(pts),5647);self.assertEqual(sum(x.get('points',0) for x in aud.values()),len(pts))
    self.assertEqual(sum(x.get('reviewed_reference',0) for x in aud.values()),276)
    self.assertEqual(len(json.loads((ROOT/'pending.json').read_text())),109)
    self.assertEqual(Counter(FAMILY[x[2]] for x in pts),{'sign':2617,'fixed':3030})
    self.assertEqual(sum(x[2]=='mobile' for x in pts),277)
    self.assertEqual(set(self.v.samples),{'sign','fixed'})
    self.assertEqual(len(set(v.tobytes()[:8000] for v in self.v.samples.values())),2)

  def test_soundd_feed_continues_same_wave_between_camera_and_paired_sign(self):
    for kind in ('fixed','mobile'):
      self.v.revoke()
      self.v.matcher=Matcher([row(kind,90),row('sign',30)])
      self.update(0,0);self.update(1,10)
      gen=self.v.request[1]
      self.v.render(np.zeros(512),True,1.1)
      self.update(2,20)  # sign now owns the HUD, but must not replay from frame zero
      self.assertEqual((self.v.matcher.hud_kind,self.v.request[1]),('sign_camera',gen))
      self.v.render(np.zeros(512),True,2.1);self.assertEqual(self.v.cursor,1024)
      self.assertIs(self.v.request[2],self.v.samples['fixed'])
    self.v.revoke()
    self.v.matcher=Matcher([row('sign',20)])
    self.update(0,0);self.update(1,10)
    self.assertIs(self.v.request[2],self.v.samples['sign'])


if __name__=='__main__':
  unittest.main()
