"""Build an offline index from the archived government dataset files. No geocoding."""
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPASS = {"北": 0, "東北": 45, "東": 90, "東南": 135, "南": 180, "西南": 225, "西": 270, "西北": 315}


def camera_road(address):
  """Classify the installation, not the vehicle's speed or a nearby intersecting highway."""
  s = unicodedata.normalize("NFKC", address).replace("臺", "台").replace(" ", "")
  if re.search(r"橋下|橋底|高架下|下方|平面|[側辅輔]車?道|側\)|慢車道|匝道|交流道口|路口|街口|巷口|與|和.{0,10}交叉", s):
    return None
  km = re.search(r"(\d+(?:\.\d+)?)\s*(?:公里|[Kk])", s)
  if re.search(r"國道(?:三(?:號)?甲|3(?:號)?甲)", s):
    return "express"
  if km and re.search(r"國道[一二三四五六八十\d]+(?:號)?|中山高速公路|福爾摩沙高速公路", s):
    return "freeway"
  if re.search(r"高速公路", s) and km:
    return "freeway"
  if km and re.search(r"快速(?:公路|道路)", s):
    return "express"
  route = re.search(r"台(\d+)([甲乙丙丁戊己庚辛]?)", s)
  if km and route:
    num = route[1]+route[2]
    if num == "61":
      k = float(km[1])
      if 2.606 <= k < 66.082 or 76 <= k <= 305.75:
        return "express"
    elif num in {"62","62甲","64","65","66","68","68甲","72","74","76","78","82","84","86","88"}:
      return "express"
  return None


def directions(s):
  axes = {"南北雙向": [0, 180], "東西雙向": [90, 270], "南北向": [0, 180], "東西向": [90, 270]}
  if s in axes:
    return axes[s]
  for pattern in (r"[東南西北]{1,2}[向往至]([東南西北]{1,2})", r"往([東南西北]{1,2})", r"([東南西北]{1,2})向"):
    m = re.fullmatch(pattern, s)
    if m and m[1] in COMPASS:
      return [COMPASS[m[1]]]
  return []  # 未知或雙向；順向/逆向是相對里程，不能當成東西南北。


def read_rows(p):
  if p.suffix == ".zip":
    with zipfile.ZipFile(p) as z:
      [name] = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.lower().endswith("manifest.csv")]
      raw = z.read(name)
  else:
    raw = p.read_bytes()
  try:
    text = raw.decode("utf-8-sig")
  except UnicodeDecodeError:
    text = raw.decode("cp950")
  return csv.DictReader(io.StringIO(text.lstrip("\ufeff"), newline=""))


def build(folder):
  rows, skipped, seen_files, audit = [], [], set(), {}
  reviewed = json.loads((ROOT/"location_refs.json").read_text())
  applied = set()
  def address_key(area, text):
    area = area.strip()
    text = text.strip().replace("臺","台").replace(" ","").removeprefix("台中市").removeprefix(area)
    if "與" in text and text.endswith("口"):
      text = text[:-1]
    return area, tuple(sorted(text.split("與")))
  # Only use a published coordinate from exactly the same district + intersection/address.
  refs = {}
  for key in ("7320-1.csv","83881-1.csv"):
    for r in read_rows(folder/key):
      if key.startswith("7320") and r["CityName"] not in ("臺中市","台中市"):
        continue
      area = r.get("RegionName") or r.get("行政區","")
      addr = r.get("Address") or r.get("設置地點","")
      try:
        lat = float(r.get("Latitude") or r.get("座標緯度",""))
        lon = float(r.get("Longitude") or r.get("座標經度",""))
      except ValueError:
        continue
      if area and addr and 23.9 < lat < 24.5 and 120.4 < lon < 121.5:
        refs.setdefault(address_key(area,addr),set()).add((lat,lon))
  for p in sorted(folder.iterdir()):
    if p.suffix.lower() not in (".csv", ".zip"):
      continue
    src = p.name.split("-")[0]
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest in seen_files:
      audit[p.name] = {"duplicate_file": True}
      continue
    seen_files.add(digest)
    counts = Counter()
    records = list(read_rows(p))
    if src == "169929":
      signs = json.loads((ROOT/"sign_review.json").read_text())
      if signs["source_sha256"] != digest:
        raise ValueError("Sign source changed; rebuild the row-keyed sign review")
    for line, r in enumerate(records, 2):
      counts["input"] += 1
      if src == "169929" and (reason := signs["skip"].get(str(line))):
        counts[reason] += 1
        if reason == "suspect_coord":
          skipped.append([p.name, line, r["公路編號"]+" "+r["起點樁號"], "警52座標異常，待查"])
        continue
      def get(*keys):
        return next((str(r[k]).strip() for k in keys if r.get(k)), "")
      item = get("取締項目", '取締項目(以"、"分隔)', "item")
      desc = get("種類", "科技執法種類")
      addr = get("Address", "取締地點", "設置地點", "location", "設置地點_範圍", "設置地點_路口或路段", "設置地點(路口或路段)")
      if src == "7320" and addr == "設置地址" and get("Latitude") == "緯度":
        counts["embedded_header"] += 1
        continue
      speed = bool(re.search("超速|測速", item))
      red = "闖紅燈" in item
      kind = ("sign" if src == "169929" else "mobile" if src == "170672" else
              "section" if src in ("126156", "172725") or "區間" in desc else
              "combined" if red and speed else "red" if red else "fixed" if speed else
              "listed" if src == "7320" else "tech")
      heading = directions(get("拍攝方向", "取締行向", "direct"))
      if src == "169929" and str(line) in signs["headings"]:
        heading = [signs["headings"][str(line)]]
      if src == "169929" and heading:
        counts["estimated_heading"] += 1
      # The national table has no enforcement-type column; do not assume every entry is speed-only.
      raw_limit = get("速限", "limit", "速限Speed")
      m = re.fullmatch(r"(\d{2,3})(?:公里)?", raw_limit)
      limit = int(m[1]) if m and int(m[1]) in range(20, 121, 5) and kind not in ("sign", "red", "tech", "listed") else 0
      lat = get("start latitude", "坐標-N-WGS84", "座標緯度", "緯度", "Latitude").split()
      lon = get("start longitude", "坐標-E-WGS84", "座標經度", "經度", "Longitude").split()
      uid = f"{src}:{line}:1"
      if uid in reviewed:
        area, address, direction, method, a, b = reviewed[uid]
        if (area,address,direction) != (get("行政區"),addr,get("取締行向","拍攝方向")):
          raise ValueError(f"Source record changed: {uid}; re-review geocode")
        if src == "13940":
          if lon != [str(b)+str(a)] or lat != [str(a)]:
            raise ValueError(f"Original format changed: {uid}")
        elif lat or lon:
          raise ValueError(f"Source has new coordinates: {uid}")
        lat,lon=[str(a)],[str(b)]
        counts["reviewed_reference"] += 1
        applied.add(uid)
      if not lat and not lon and src == "170672":
        matches = refs.get(address_key(get("行政區"),addr),set())
        if len(matches) == 1:
          a,b = next(iter(matches))
          lat,lon = [str(a)], [str(b)]
          counts["same_address_reference"] += 1
      try:
        if not lat or len(lat) != len(lon):
          raise ValueError
        coords = [(float(a), float(b)) for a, b in zip(lat, lon)]
        if not all(20 <= a <= 27 and 117 <= b <= 124 for a, b in coords):
          raise ValueError
      except ValueError:
        counts["excluded"] += 1
        skipped.append([p.name, line, addr, "缺少或無法驗證座標"])
        continue
      for a, b in coords:
        road = camera_road(addr) if kind != "sign" else None
        rows.append([round(a, 7), round(b, 7), kind, heading, limit, src] + ([road] if road else []))
        if road:
          counts[road] += 1
        counts["points"] += 1
    audit[p.name] = dict(counts)
  if applied != set(reviewed):
    raise ValueError("Reference table does not match this source snapshot")
  return rows, audit, skipped


if __name__ == "__main__":
  if len(sys.argv) != 3 or Path(sys.argv[2]).resolve() == ROOT.resolve():
    raise SystemExit("Historical CSV build: build_points.py <raw folder> <separate output folder>. "
                     "For the current car index use: python3 -m sunnypilot.tw_alerts.import_workbook <xlsx>")
  out = Path(sys.argv[2])
  rows, audit, skipped = build(Path(sys.argv[1]))
  (out / "points.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
  (out / "import-report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
  (out / "pending.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=1), encoding="utf-8")
  print(len(rows), "points;", len(skipped), "unlocated rows;", Counter(r[2] for r in rows))
