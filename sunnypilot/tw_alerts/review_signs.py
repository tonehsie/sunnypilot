"""Build-time only: reconcile the v2 sign review with the original signs and official mileposts."""
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

from .build_points import ROOT, read_rows


def sign_key(r):
  return (r["公路編碼"], r["起點樁號"], r["牌面方向"], float(r["坐標-N-WGS84"]), float(r["坐標-E-WGS84"]))


def metres(s):
  m = re.fullmatch(r"(\d+)K\+(\d{1,3})", s.strip(), re.I)
  return int(m[1])*1000+int(m[2]) if m else None


def mileposts(path):
  with zipfile.ZipFile(path) as z:
    entries = [n for n in z.namelist() if n.lower().endswith(".csv") and n.rsplit("/",1)[-1] not in
               {"file.csv","manifest.csv","schema-file.csv"}]
    [name] = entries
    raw = z.read(name)
  try:
    source = raw.decode("utf-8-sig")
  except UnicodeDecodeError:
    source = raw.decode("cp950")
  out = defaultdict(lambda: defaultdict(set))
  for r in csv.DictReader(io.StringIO(source)):
    k = metres(r["起點樁號"])
    if k is not None and r["性質"] == "指45":
      out[r["公路編碼"]][k].add((float(r["坐標-N-WGS84"]),float(r["坐標-E-WGS84"])))
  return {road:(sorted(marks), marks) for road,marks in out.items()}


def corroborated(r, roads):
  # These are estimates of travel direction, never camera positions. Check a
  # short, nearly straight span between physical mileposts on the same route.
  road,mark,face,lat,lon = sign_key(r)
  if not r["校正優先級"].startswith(("OK_", "P4_")) or face not in ("順向","逆向") or not r["heading_deg"]:
    return None
  k = metres(mark)
  if k is None or road not in roads:
    return None
  ks, marks = roads[road]
  i = bisect_left(ks,k)
  j = i+1 if i < len(ks) and ks[i] == k else i
  if i == 0 or j == len(ks):
    return None
  lo,hi = ks[i-1],ks[j]
  if not 30 <= hi-lo <= 1100:
    return None
  estimates = []
  for a,b in marks[lo]:
    for c,d in marks[hi]:
      scale = 111320*math.cos(math.radians(lat))
      dx,dy = (d-b)*scale,(c-a)*111320
      if not .96 <= math.hypot(dx,dy)/(hi-lo) <= 1.05:
        return None
      part = (k-lo)/(hi-lo)
      if math.hypot((lon-b)*scale-dx*part,(lat-a)*111320-dy*part) > 45:
        return None
      estimates.append((math.degrees(math.atan2(dx,dy))+(180 if face == "逆向" else 0))%360)
  h = int(r["heading_deg"])
  if 0 <= h < 360 and estimates and all(abs((h-b+180)%360-180) <= 15 for b in estimates):
    return h
  return None


def compile_review(raw, v2, posts):
  rows = list(read_rows(raw))
  review = list(read_rows(v2))
  by_key = {sign_key(r):r for r in review}
  if len(by_key) != len(review) or set(by_key) != {sign_key(r) for r in rows}:
    raise ValueError("v2 and official signs are not an exact match by route, mileage, face and coordinates")
  roads,seen,skip,headings = mileposts(posts),set(),{},{}
  for line, row in enumerate(rows,2):
    key = sign_key(row)
    r = by_key[key]
    if key in seen:
      skip[line] = "duplicate"
    elif r["校正優先級"].startswith("P1_"):
      skip[line] = "suspect_coord"
    else:
      heading = corroborated(r, roads)
      if heading is not None:
        headings[line] = heading
    seen.add(key)
  return {"source_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),
          "v2_sha256":hashlib.sha256(v2.read_bytes()).hexdigest(),
          "mileposts_sha256":hashlib.sha256(posts.read_bytes()).hexdigest(), "skip":skip, "headings":headings}


if __name__ == "__main__":
  table = compile_review(*map(Path,sys.argv[1:]))
  (ROOT/"sign_review.json").write_text(json.dumps(table,indent=1)+"\n")
