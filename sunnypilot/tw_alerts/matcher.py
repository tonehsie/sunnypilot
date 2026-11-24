"""Offline approach tracker. No car-control or network dependencies."""
import math
from collections import defaultdict
from typing import NamedTuple

ENFORCEMENT = frozenset(("fixed", "mobile", "section", "combined", "red", "tech", "listed"))
FAMILY = {"sign": "sign", **dict.fromkeys(ENFORCEMENT, "fixed")}
CAMERA_BEFORE = {"ordinary": 300., "express": 600., "freeway": 800.}
CAMERA_AFTER = 10.
SIGN_BEFORE, SIGN_AFTER = 10., 300.
EPS = 1e-6


class Site(NamedTuple):
  lat: float
  lon: float
  kind: str
  headings: tuple
  limit: int
  road: str


def offset(lat, lon, a, b):
  return (b - lon) * 111320 * math.cos(math.radians(lat)), (a - lat) * 111320


def angle(a, b):
  return abs((a - b + 180) % 360 - 180)


def relative(lat, lon, site, course):
  x, y = offset(lat, lon, site.lat, site.lon)
  s, c = math.sin(math.radians(course)), math.cos(math.radians(course))
  return x*s + y*c, abs(x*c - y*s)


class Matcher:
  def __init__(self, rows):
    self.grid = defaultdict(list)
    for a, b, kind, dirs, limit, _, *road in rows:
      if kind not in FAMILY or not (20 <= a <= 27 and 117 <= b <= 124):
        raise ValueError("invalid point")
      road_class = road[0] if road else "ordinary"
      if road_class not in CAMERA_BEFORE:
        raise ValueError("invalid road class")
      site = Site(a, b, kind, tuple(dirs), limit, road_class)
      self.grid[int(a*100), int(b*100)].append(site)
    self.passed = {}
    self.reset()

  def reset(self):
    self.active = None
    self.encounters = {}  # keep independent windows even while a sign owns the bell
    self.paired_signs = set()
    self.trusted_course = None
    self.course = 0.
    self.distance = 0.
    self.last = self.origin = None

  def nearby(self, lat, lon, radius):
    dlat = radius/111320
    dlon = dlat/math.cos(math.radians(lat))
    for i in range(int((lat-dlat)*100), int((lat+dlat)*100)+1):
      for j in range(int((lon-dlon)*100), int((lon+dlon)*100)+1):
        yield from self.grid.get((i,j), ())

  @staticmethod
  def on_approach(site, course):
    return not site.headings or any(angle(course,h) <= 55 for h in site.headings)

  @staticmethod
  def width(site):
    return 26 if site.headings else 15

  def sign_has_point(self, sign, lat, lon, course):
    # Measure from the SIGN, not from the current fix: a driver 10 m before a
    # sign may be 310 m from a point at the far end of its 300 m zone.
    for site in self.nearby(sign.lat, sign.lon, SIGN_AFTER + 45):
      if site.kind not in ENFORCEMENT or not self.on_approach(site, course):
        continue
      along, side = relative(sign.lat,sign.lon,site,course)
      _, road_side = relative(lat,lon,site,course)
      if -EPS <= along <= SIGN_AFTER + EPS and side <= self.width(site) and road_side <= self.width(site):
        return True
    return False

  @property
  def sound_family(self):
    if self.active is None:
      return None
    return "fixed" if self.active in self.paired_signs else FAMILY[self.active.kind]

  @property
  def hud_kind(self):
    if self.active is None:
      return None
    return "sign_camera" if self.active in self.paired_signs else self.active.kind

  def fix(self, now, lat, lon, speed, bearing, accuracy, bearing_accuracy, *, modem_track=False):
    moving = speed >= 2
    if not (all(map(math.isfinite, (now, lat, lon, speed, accuracy))) and 20 <= lat <= 27 and 117 <= lon <= 124
            and 0 <= speed <= 60 and (0 < accuracy <= 20 or (modem_track and accuracy == 0)) and
            (not moving or (0 <= bearing < 360 and (modem_track or 0 <= bearing_accuracy <= 35)))):
      self.reset()
      return None
    old = self.last
    if old is None or not 0 < now-old[0] <= 2:
      self.reset()
      self.last = self.origin = (now, lat, lon)
      return None
    self.last = (now, lat, lon)
    if math.hypot(*offset(old[1], old[2], lat, lon)) > speed*(now-old[0])*2 + max(15, accuracy):
      self.reset()
      return None
    origin = self.origin
    dx, dy = offset(origin[1], origin[2], lat, lon)
    # Unknown modem precision is not a perfect fix. Require actual displacement
    # corroborating the modem's course before acquiring a point.
    progress = math.hypot(dx, dy) >= (12. if modem_track else max(3, accuracy*.6))
    if progress:
      self.origin = self.last
      if moving and angle(math.degrees(math.atan2(dx,dy)), bearing) > 45:
        self.reset()
        return None
      if moving:
        self.trusted_course = bearing

    self.passed = {site:p for site,p in self.passed.items() if now-p[0] < 90}
    for site, (axis, dist) in list(self.encounters.items()):
      is_sign = site.kind == "sign"
      if is_sign and dist <= 0:
        # Beyond a sign, integrate forward movement along the local road rather
        # than a straight chord from the sign; stationary jitter isn't distance.
        if moving and angle(bearing, axis) > 55:
          del self.encounters[site]
          continue
        x, y = offset(old[1], old[2], lat, lon)
        course = bearing if moving else axis
        rad = math.radians(course)
        if speed >= .5:
          dist -= x*math.sin(rad) + y*math.cos(rad)
        axis = course
      else:
        dist, side = relative(lat, lon, site, axis)
        if side > 45 or (moving and angle(bearing,axis) > 60):
          del self.encounters[site]
          continue
      if dist <= -(SIGN_AFTER if is_sign else CAMERA_AFTER) + EPS:
        self.passed[site] = (now, axis)
        del self.encounters[site]
      else:
        self.encounters[site] = (axis, dist)

    # Once course is corroborated, check every fresh moving fix, not just
    # every 12 m: the entire pre-sign window can lie between two 1 Hz fixes.
    self.paired_signs.intersection_update(self.encounters)
    if moving and self.trusted_course is not None and angle(bearing,self.trusted_course) <= 45:
      for site in self.nearby(lat, lon, max(CAMERA_BEFORE.values()) + 45):
        if site in self.encounters or (site in self.passed and angle(self.passed[site][1],bearing) < 75):
          continue
        if not self.on_approach(site,bearing):
          continue
        ahead, side = relative(lat,lon,site,bearing)
        width = self.width(site)
        if side > width:
          continue
        if site.kind == "sign":
          if ahead > SIGN_BEFORE + EPS:
            continue
          if ahead < 0:
            prev, prev_side = relative(old[1],old[2],site,bearing)
            if prev < 0 or prev_side > width or ahead <= -SIGN_AFTER + EPS:
              continue
          if self.sign_has_point(site,lat,lon,bearing):
            self.paired_signs.add(site)
        elif not 0 <= ahead <= CAMERA_BEFORE[site.road] + EPS:
          continue
        self.encounters[site] = (bearing, ahead)
      # A gentle curve can hide a downstream point from the initial tangent.
      # Only upgrade if driven distance + distance still ahead is inside the zone.
      for sign, (_,dist) in self.encounters.items():
        if sign.kind != "sign" or dist >= 0 or sign in self.paired_signs:
          continue
        for point in self.encounters:
          if point.kind not in ENFORCEMENT or not self.on_approach(point,bearing):
            continue
          ahead,side = relative(lat,lon,point,bearing)
          if 0 <= ahead and ahead-dist <= SIGN_AFTER + EPS and side <= self.width(point):
            self.paired_signs.add(sign)
            break
    # A sign owns its zone. An unrelated nearby point must not relabel it.
    # Outside sign windows, use the normal standalone enforcement windows.
    self.active = min(self.encounters, key=lambda s:(0 if s in self.paired_signs else 1 if s.kind == "sign" else 2, s != self.active,
                      abs(self.encounters[s][1])), default=None)
    if self.active is not None:
      self.course, self.distance = self.encounters[self.active]
    return self.active
