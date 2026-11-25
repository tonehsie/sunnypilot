"""Build the offline reminder index from the reviewed seven-column CSV (host only)."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .matcher import CAMERA_BEFORE, FAMILY, Matcher

HEADER = ['緯度','經度','類型','行車方向角','速限（km/h）','來源資料集編號','道路等級']
ROOT = Path(__file__).parent


def build(source, out):
  points, limits = {}, {}
  count = 0
  exact = set()
  with source.open(encoding='utf-8-sig', newline='') as f:
    rows = csv.reader(f, strict=True)
    if next(rows) != HEADER:
      raise ValueError('unexpected header')
    for line,row in enumerate(rows,2):
      lat,lon,kind,h,s,src,road = row
      a,b,dirs,limit = float(lat),float(lon),json.loads(h),int(s)
      if not (20<=a<=27 and 117<=b<=124 and kind in FAMILY and isinstance(dirs,list) and len(dirs)<=2 and
              all(type(v) is int and 0<=v<360 for v in dirs) and len(set(dirs))==len(dirs) and
              (limit==0 or limit in range(20,121,5)) and src.isascii() and src.isdigit() and
              (kind=='sign')==(src=='169929') and (not road or road in CAMERA_BEFORE)):
        raise ValueError(f'invalid point at line {line}')
      key=(a,b,kind,tuple(dirs),src,road)
      points.setdefault(key,[a,b,kind,dirs,limit,src]+([road] if road else []))
      limits.setdefault(key,set()).add(limit)
      exact.add((key,limit))
      count+=1
  if not points: raise ValueError('empty input')
  conflicts=[]
  for key,vals in limits.items():
    if len(vals)>1:
      points[key][4]=0
      conflicts.append({'point':points[key], 'input_limits':sorted(vals)})
  data=list(points.values())
  Matcher(data)
  blob=json.dumps(data,separators=(',',':'),allow_nan=False).encode()
  report=dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input=count,
              active=len(data),exact_duplicates=count-len(exact),limit_conflicts=conflicts,
              points_sha256=hashlib.sha256(blob).hexdigest(),by_kind=dict(Counter(x[2] for x in data)),
              direction_counts=dict(Counter(len(x[3]) for x in data)))
  (out/'points.json').write_bytes(blob)
  (out/'csv-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
  import argparse
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source',type=Path)
  parser.add_argument('--out',type=Path,default=ROOT)
  args=parser.parse_args()
  build(args.source,args.out)
