"""Rebuild the two offline 48 kHz mono bell patterns; no speech or TTS."""
import wave
from pathlib import Path

import numpy as np

RATE = 48000
# (bell frequencies, seconds between note starts). Fixed gets a single long bell.
PATTERNS = {"sign": ((805, 570), .23), "fixed": ((485,), .25)}


def chime(tones, step):
  duration = .32 if len(tones) > 1 else .48
  t = np.arange(round(RATE * duration)) / RATE
  env = (1 - np.exp(-t / .004)) * np.exp(-t / (duration * .25)) * np.clip((duration - t) / .035, 0, 1)
  start_step = round(RATE * step)
  out = np.zeros(len(t) + (len(tones) - 1) * start_step)
  for i, hz in enumerate(tones):
    phase = 2 * np.pi * hz * t
    out[i*start_step:i*start_step+len(t)] += env * (np.sin(phase) + .24*np.sin(2*phase) + .08*np.sin(3*phase))
  # +3 dB from 15500. Leaves headroom even with the optional speed-limit chime.
  return np.rint(out * (21900 / np.max(np.abs(out)))).astype("<i2")


def main():
  # Preserve the existing asset paths for installed readers.
  root = Path(__file__).resolve().parent / "voice"
  for key, spec in PATTERNS.items():
    with wave.open(str(root / (key + ".wav")), "wb") as f:
      f.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
      f.writeframes(chime(*spec).tobytes())


if __name__ == "__main__":
  main()
