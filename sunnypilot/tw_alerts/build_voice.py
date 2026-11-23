"""Build-time only: espeakng-loader 0.2.4, numpy. No TTS or external services on the car."""
import ctypes as C
import wave
from pathlib import Path

import espeakng_loader
import numpy as np

ROOT = Path(__file__).resolve().parent
PHRASES = {"mobile": ("移动式测速路段。", (530, 650, 790)),
           "sign": ("警五十二，注意速限。", (660, 530)),
           "fixed": ("前方执法设置点。", (580,))}


def make():
  lib = C.CDLL(espeakng_loader.get_library_path())
  rate = lib.espeak_Initialize(2, 0, str(Path(espeakng_loader.get_data_path()).parent).encode(), 0)
  if rate <= 0 or lib.espeak_SetVoiceByName(b"cmn") != 0:
    raise RuntimeError("Mandarin voice unavailable")
  lib.espeak_SetParameter(1, 165, 0)  # espeakRATE
  frames = []
  @C.CFUNCTYPE(C.c_int, C.POINTER(C.c_short), C.c_int, C.c_void_p)
  def capture(samples, count, _):
    if samples and count > 0:
      frames.append(np.ctypeslib.as_array(samples, shape=(count,)).copy())
    return 0
  lib.espeak_SetSynthCallback(capture)
  lib.espeak_Synth.argtypes = [C.c_void_p,C.c_size_t,C.c_uint,C.c_int,C.c_uint,C.c_uint,C.c_void_p,C.c_void_p]
  out = ROOT / "voice"
  out.mkdir(exist_ok=True)
  for name, (text, pitches) in PHRASES.items():
    frames.clear()
    data=text.encode()
    if lib.espeak_Synth(data, len(data)+1, 0, 1, 0, 1, None, None) != 0:
      raise RuntimeError(name)
    raw = np.concatenate(frames).astype(float)
    active = np.flatnonzero(abs(raw)>60)
    if len(active)<rate*.08:
      raise RuntimeError("silent voice "+name)
    raw = raw[max(0,active[0]-rate//40):active[-1]+1+rate//10]
    x = np.interp(np.arange(int(len(raw)*48000/rate))*rate/48000, np.arange(len(raw)),raw)
    t = np.arange(5760)/48000
    prefix = [np.concatenate((6400*np.hanning(len(t))*np.sin(2*np.pi*hz*t), np.zeros(2160))) for hz in pitches]
    pcm = np.concatenate([*prefix, np.zeros(3840), x*(16000/max(abs(x)))]).astype("<i2")
    with wave.open(str(out/(name+".wav")),"wb") as w:
      w.setparams((1,2,48000,0,"NONE","not compressed"))
      w.writeframes(pcm.tobytes())
  lib.espeak_Terminate()


if __name__ == "__main__":
  make()
