import ast
import json
import math
import unittest
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from .audio import GAP, ROOT, Voice
from .matcher import FAMILY, Matcher


def row(kind, distance, lateral=0, dirs=(0,)):
  return [24+distance/111320,121+lateral/(111320*math.cos(math.radians(24))),kind,list(dirs),0,'test']


def fix(m,t,y,x=0,speed=10,course=0,acc=4):
  return m.fix(t,24+y/111320,121+x/(111320*math.cos(math.radians(24))),speed,course,acc,5)


class Tracking(unittest.TestCase):
  def test_pass_boundary_and_no_replay(self):
    for kind in FAMILY:
      with self.subTest(kind=kind):
        m=Matcher([row(kind,50)])
        self.assertIsNone(fix(m,0,0))
        for t,y in enumerate([10,20,30,40,49,50,55,59.9],1):
          self.assertEqual(fix(m,t,y).kind,kind)
        self.assertIsNone(fix(m,9,60))
        self.assertIsNone(fix(m,10,65))

  def test_priorities_and_fallback_to_next_unpassed_site(self):
    m=Matcher([row('fixed',70),row('sign',60),row('mobile',50)])
    fix(m,0,0)
    self.assertEqual(fix(m,1,10).kind,'mobile')
    for i in range(2,6):self.assertEqual(fix(m,i,i*10).kind,'mobile')
    self.assertEqual(fix(m,6,60).kind,'fixed')  # sign at the car is no longer ahead
    m=Matcher([row('section',90),row('sign',130),row('mobile',165)])
    fix(m,0,0);self.assertEqual(fix(m,1,10).kind,'sign')
    for i in range(2,4):active=fix(m,i,i*10)
    self.assertEqual(active.kind,'mobile')  # preempts the sign

  def test_separate_encounters(self):
    m=Matcher([row('mobile',35),row('sign',80),row('fixed',135)])
    kinds=[(s.kind if (s:=fix(m,i,i*10)) else None) for i in range(17)]
    self.assertEqual([v for i,v in enumerate(kinds) if i==0 or v!=kinds[i-1]],
                     [None,'mobile','sign','fixed',None])

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
    self.v=Voice()
    self.v.matcher=Matcher([row('mobile',50)])
    class SM(dict):pass
    g=NS(hasFix=True,latitude=24,longitude=121,speed=10,bearingDeg=0,bearingAccuracyDeg=5,horizontalAccuracy=4)
    self.sm=SM(g=g)
    self.sm.alive={'g':True};self.sm.valid={'g':True};self.sm.updated={'g':True};self.sm.recv_time={'g':0.}

  def update(self,i,m,permitted=True):
    self.sm.recv_time['g']=i;self.sm['g'].latitude=24+m/111320
    with patch(__package__+'.audio.notify'):
      self.v.update(self.sm,'g',i,permitted)

  def test_loop_without_restarting_on_each_fix(self):
    self.update(0,0);self.update(1,10)
    key=self.v.request[1];sample=self.v.samples['mobile'];buf=np.zeros(len(sample)+GAP+len(sample))
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
    self.sm['g'].hasFix=False;self.update(5,45)
    self.assertIsNone(self.v.request)

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
    self.assertEqual(Counter(FAMILY[x[2]] for x in pts),{'mobile':277,'sign':2617,'fixed':2753})
    self.assertEqual(set(self.v.samples),{'mobile','sign','fixed'})
    self.assertEqual(len(set(v.tobytes()[:8000] for v in self.v.samples.values())),3)


if __name__=='__main__':
  unittest.main()
