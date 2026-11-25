"""Offline build tool for the reviewed Taiwan workbook. Never imported by soundd."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from .build_points import ROOT, camera_road

SHEET = "重校正總表"
KINDS = {"警52預告標誌":"sign", "固定測速":"fixed", "移動式取締地點":"mobile", "闖紅燈":"red",
         "測速及闖紅燈":"combined", "區間測速":"section", "科技執法":"tech", "全國測速設置點":"listed"}
FIELDS = {"新序號","校正優先級","類型","緯度","經度","heading_deg","heading2_deg","速限","地址或里程","資料集編號","點位識別碼"}


def integer(value, low, high):
  if type(value) not in (float,int) or not math.isfinite(value) or int(value) != value or not low <= value <= high:
    raise ValueError(f"invalid integer {value!r}")
  return int(value)


def compile_rows(records):
  points, quarantined, seen = [], [], set()
  for line,r in enumerate(records,2):
    try:
      number = integer(r["新序號"],1,10**7)
      if number in seen:
        raise ValueError("duplicate serial")
      seen.add(number)
      kind = KINDS[r["類型"]]
      priority = r["校正優先級"].split("_",1)[0]
      if priority not in {"P1","P2","P3","P4","P5","P6","OK"}:
        raise ValueError("unknown review priority")
      if priority == "P1":
        quarantined.append([number,r["點位識別碼"],r["地址或里程"]])
        continue
      a,b = r["緯度"],r["經度"]
      if not (type(a) in (int,float) and type(b) in (int,float) and 20<=a<=27 and 117<=b<=124):
        raise ValueError("invalid Taiwan coordinates")
      h,h2 = r["heading_deg"],r["heading2_deg"]
      if h is None and h2 is not None:
        raise ValueError("secondary direction without primary")
      dirs = [integer(v,0,359) for v in (h,h2) if v is not None]
      if len(dirs)==2 and (dirs[1]-dirs[0])%360 != 180:
        raise ValueError("expected opposite directions")
      if (priority in {"P2","P3"}) != (not dirs):
        raise ValueError("review priority and directions disagree")
      m = re.fullmatch(r"(\d{2,3})(?:公里)?",str(r["速限"]))
      limit = int(m[1]) if m and int(m[1]) in range(20,121,5) and kind not in {"listed","red","tech","sign"} else 0
      src = str(r["資料集編號"])
      if (src == "公路局警52") != (kind == "sign") or (kind != "sign" and not src.isascii()) or (kind != "sign" and not src.isdigit()):
        raise ValueError("unexpected dataset")
      src = "169929" if kind == "sign" else src
      road = None if kind == "sign" else camera_road(r["地址或里程"])
      point = [round(a,7),round(b,7),kind,dirs,limit,src]+([road] if road else [])
      # Preserve original source order for collocated camera ties, not the Excel review priority.
      order = (number,0) if kind == "sign" else tuple(map(int,r["點位識別碼"].split(":")[1:]))
      if kind != "sign" and (len(order)!=2 or not r["點位識別碼"].startswith(src+":")):
        raise ValueError("unexpected source ID")
      points.append(((src,*order),point))
    except (AttributeError,KeyError,TypeError,ValueError) as e:
      raise ValueError(f"{SHEET} row {line}: {e}") from e
  if seen != set(range(1,len(seen)+1)) or not points:
    raise ValueError("missing serials or empty worksheet")
  return [point for _,point in sorted(points,key=lambda p:p[0])],quarantined


def build_from_workbook(path, out=ROOT):
  from openpyxl import load_workbook  # host only, never installed or needed on the car
  w = load_workbook(path, read_only=True, data_only=False, keep_links=False)
  try:
    rows = iter(w[SHEET].values)
    header = next(rows)
    if len(set(header)) != len(header) or not FIELDS.issubset(header):
      raise ValueError("missing/duplicate workbook headers")
    points,excluded = compile_rows(dict(zip(header,r,strict=True)) for r in rows)
  finally:
    w.close()
  blob = json.dumps(points,separators=(",",":"),allow_nan=False).encode()
  report = {"sheet":SHEET,"workbook_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "points_sha256":hashlib.sha256(blob).hexdigest(),"input":len(points)+len(excluded),
            "active":len(points),"by_kind":dict(sorted(Counter(p[2] for p in points).items())),
            "direction_counts":dict(sorted(Counter(len(p[3]) for p in points).items())),
            "quarantined":excluded}
  out = Path(out)
  (out/"points.json").write_bytes(blob)
  (out/"workbook-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
  return report


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("workbook",type=Path)
  parser.add_argument("--out",type=Path,default=ROOT)
  args = parser.parse_args()
  print(build_from_workbook(args.workbook,args.out)["active"],"active points")
