"""Read-only, six-second 3X check. Does not play sounds or print coordinates."""
import time
from pathlib import Path

from .audio import Chimes, GPS_SERVICES, gps_input


def main():
  from cereal import messaging
  from openpilot.common.params import Params
  from openpilot.system.hardware import HARDWARE

  params = Params()
  print("機型:", HARDWARE.get_device_type(), "靜音:", params.get_bool("QuietMode"),
        "提醒停用檔:", Path("/data/disable_tw_road_voice").exists())
  print("可載入音效:", ", ".join(sorted(Chimes().samples)))
  sm = messaging.SubMaster([*GPS_SERVICES, "selfdriveState"])
  stats = {s: [0,0,0] for s in GPS_SERVICES}
  until = time.monotonic() + 6
  while time.monotonic() < until:
    sm.update(100)
    for s in GPS_SERVICES:
      if sm.updated[s]:
        g = sm[s]
        stats[s][0] += 1
        stats[s][1] += int(g.hasFix)
        stats[s][2] += int(sm.valid[s] and gps_input(g,s) is not None)
  for s,(recv,fix,usable) in stats.items():
    print(f"{s}: 收到 {recv} 筆，有定位 {fix} 筆，可供比對 {usable} 筆")
  print("行車狀態訊號有效:", sm.alive["selfdriveState"] and sm.valid["selfdriveState"])
  print("熄火時沒有定位可能正常。這項檢查不需要移動車輛，也不會播音。")


if __name__ == "__main__":
  main()
