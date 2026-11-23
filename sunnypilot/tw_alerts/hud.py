"""Best-effort, nonblocking local notification. UI absence must never affect sound."""
import math
import os
import socket
import struct
import time

LABELS = {"sign":"警52 測速取締路段","fixed":"固定測速","mobile":"移動測速路段","section":"區間測速",
          "combined":"測速及闖紅燈","red":"闖紅燈照相","tech":"科技執法","listed":"執法提醒"}
KINDS = tuple(LABELS)
PACKET = struct.Struct("!dBBH")
ADDRESS = "\0sp-tw-road-v1-" + str(os.getuid())


def notify(kind, limit, meters, now):
  try:
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
      s.setblocking(False)
      s.sendto(PACKET.pack(now,255 if kind is None else KINDS.index(kind),limit,min(65535,round(meters))), ADDRESS)
  except OSError:
    pass


class Hud:
  def __init__(self):
    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
      self.sock.setblocking(False)
      self.sock.bind(ADDRESS)
    except OSError:
      self.sock.close()
      raise
    self.event = None

  def text(self, now, speed, permitted):
    for _ in range(4):
      try:
        data = self.sock.recv(PACKET.size+1)
      except (BlockingIOError,OSError):
        break
      if len(data)==PACKET.size:
        t,k,lim,m = PACKET.unpack(data)
        if math.isfinite(t) and 0<=now-t<.7:
          if k==255:
            self.event=None
          elif k<len(KINDS) and lim<=120 and m<=500:
            self.event=(t,k,lim,m)
    if not permitted:
      self.event=None
    if self.event is None:
      return None
    t,k,lim,m=self.event
    if not 0<=now-t<3 or not math.isfinite(speed):
      self.event=None
      return None
    dist=max(0,m-max(0,speed)*(now-t))
    return LABELS[KINDS[k]], f"約 {round(dist/10)*10} m" + (f"  ·  {lim} km/h" if lim else "")
