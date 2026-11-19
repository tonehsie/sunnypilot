"""Bake smooth Traditional Chinese BMFont atlases; no network or fontTools required.

OpFont is from dragonpilot (IBM/Google/Rick Lan, SIL OFL 1.1). Latin glyphs
retain Inter outlines. Existing Inter/Unifont assets are never overwritten.
"""
from pathlib import Path
import json
import math
import re

from PIL import Image, ImageDraw, ImageFont

FONT_SIZE = 48
PADDING = 3
ATLAS_WIDTH = 2048
# Supplement catalog glyphs with vehicle/settings terminology; not all Unicode.
EXTRA_TC = """
學習維持輔助轉向扭矩方向盤回正角度偏移剛性側傾延遲摩擦加速度曲線
原生預設固定即時手動自動控制啟動啟用取消停用停止暫停恢復退出保持
車輛車型車道變換彎道路口低速高速公路市區公里小時每秒牛頓米駕駛
煞車油門踏板主開關按鈕長按短按單擊雙擊警示警告提示聲音音量靜音
溫度處理器記憶體儲存空間剩餘使用容量錄影影像檔案刪除下載批次日期
時間時區臺灣台灣連線網路衛星訊號定位精確有效無效可用不可用資料
校正校準完成進度重設重新啟動關機設定軟體版本分支更新安裝卸載選擇
確認返回取消儲存開啟關閉允許禁止支援不支援實驗模型神經網路傳統
人工智慧扭力輸出上限下限最大最小保護機制介入接管駕駛監控注意力
超速速限地圖目前目標實際增加減少降低提高平滑平順柔和強度阻力
左側右側前方後方靠左靠右中間置中儀表側欄顯示畫面主畫面亮度字型
字體繁體中文簡體英文語言粗體細體一般同步連續單次音效安全距離
導航目的地路徑行程行車紀錄裝置電池電壓電流風扇散熱負載頻率訊息
已未有無是是否正常異常錯誤故障可否需要請勿必須立即可稍後再試
全時全速域開車停車路試測試日夜模式白紅黃綠藍黑灰顏色樣式清晰
軟硬韌體識別證書授權隱私權條款內容說明詳情提示最佳適合穩定狀態
？！（）「」『』：；、。，％～－／℃°
"""


def _is_cjk(codepoint):
  return 0x2E80 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF or 0xFF00 <= codepoint <= 0xFFEF


def required_codepoints(font_dir):
  ui = font_dir.parents[1] / "ui" / "translations"
  text = (ui / "app_zh-CHT.po").read_text(encoding="utf-8")
  labels = json.loads((ui / "languages.json").read_text(encoding="utf-8"))
  text += EXTRA_TC + "".join(labels)
  # Native emoji rendering handles supplementary emoji such as the fire icon.
  original = {int(cp) for cp in re.findall(r"^char id=(\d+)", (font_dir / "unifont.fnt").read_text(), re.M) if 32 <= int(cp) <= 0xFFFF}
  return sorted(original | set(range(32, 127)) | {ord(c) for c in text if 127 <= ord(c) <= 0xFFFF})


def _pixel_font(path, size=FONT_SIZE):
  # BMFont/raylib sizes are pixel height (ascent+descent), unlike Pillow's em size.
  em_size = size
  while em_size > 1:
    font = ImageFont.truetype(str(path), em_size)
    if sum(font.getmetrics()) <= size:
      return font
    em_size -= 1
  raise RuntimeError(f"Invalid metrics: {path}")


def _mask_signature(font, char):
  mask = font.getmask(char)
  return mask.size, bytes(mask)


def _glyphs(font_dir, weight, codepoints):
  inter = _pixel_font(font_dir / f"Inter-{weight}.ttf")
  chinese = _pixel_font(font_dir / f"OpFont-{weight}.otf")
  fallback = _pixel_font(font_dir / f"SunnyTCFallback-{weight}.otf")
  missing_inter = _mask_signature(inter, "\uffff")
  missing_chinese = _mask_signature(chinese, "\uffff")
  missing_fallback = _mask_signature(fallback, "\uffff")
  glyphs = []
  for cp in codepoints:
    char = chr(cp)
    candidates = ([(chinese, missing_chinese), (fallback, missing_fallback)] if _is_cjk(cp) else
                  [(inter, missing_inter), (chinese, missing_chinese)])
    selected = next((font for font, missing in candidates if _mask_signature(font, char) != missing), None)
    if selected is None:
      raise RuntimeError(f"Unsupported required glyph U+{cp:04X} {char!r}")
    x0, y0, x1, y1 = selected.getbbox(char, anchor="ls")
    width, height = max(x1-x0, 1), max(y1-y0, 1)
    bitmap = Image.new("L", (width, height))
    ImageDraw.Draw(bitmap).text((-x0, -y0), char, font=selected, fill=255, anchor="ls")
    # A common baseline aligns CJK with Latin and keeps pixels inside a 48px line.
    if height > FONT_SIZE:
      raise RuntimeError(f"Glyph U+{cp:04X} exceeds line height: {height}")
    # Preserve complete inherited glyphs (e.g. Thai descenders), never crop them.
    offset_y = min(max(39 + y0, 0), FONT_SIZE - height)
    glyphs.append(dict(id=cp, bitmap=bitmap, width=width, height=height,
                       xoffset=x0, yoffset=offset_y, xadvance=round(selected.getlength(char))))
  return glyphs


def _atlas(font_dir, weight, codepoints):
  glyphs = _glyphs(font_dir, weight, codepoints)
  x = y = row_height = PADDING
  for glyph in glyphs:
    if x + glyph["width"] + PADDING > ATLAS_WIDTH:
      x = PADDING
      y += row_height + PADDING
      row_height = 0
    glyph.update(x=x, y=y)
    x += glyph["width"] + PADDING
    row_height = max(row_height, glyph["height"])
  used_height = y + row_height + PADDING
  height = 2 ** math.ceil(math.log2(used_height))
  if height > 2048:
    raise RuntimeError(f"Atlas too large ({ATLAS_WIDTH}x{height}); split instead of increasing GPU memory")
  atlas = Image.new("RGBA", (ATLAS_WIDTH, height), (255, 255, 255, 0))
  name = f"TraditionalChinese-{weight}"
  lines = [
    f'info face="{name}" size=-{FONT_SIZE} bold={int(weight=="Bold")} italic=0 charset="" unicode=1 stretchH=100 smooth=1 aa=1 padding=0,0,0,0 spacing=0,0 outline=0',
    f'common lineHeight={FONT_SIZE} base=39 scaleW={ATLAS_WIDTH} scaleH={height} pages=1 packed=0 alphaChnl=0 redChnl=4 greenChnl=4 blueChnl=4',
    f'page id=0 file="{name}.png"',
    f'chars count={len(glyphs)}',
  ]
  for glyph in glyphs:
    bitmap = Image.new("RGBA", glyph["bitmap"].size, (255, 255, 255, 0))
    bitmap.putalpha(glyph["bitmap"])
    atlas.paste(bitmap, (glyph["x"], glyph["y"]))
    lines.append(('char id={id} x={x} y={y} width={width} height={height} xoffset={xoffset} yoffset={yoffset} xadvance={xadvance} page=0 chnl=15').format(**glyph))
  atlas.save(font_dir / f"{name}.png")
  (font_dir / f"{name}.fnt").write_text("\n".join(lines)+"\n", encoding="utf-8")
  return dict(weight=weight, glyphs=len(glyphs), width=ATLAS_WIDTH, height=height, rgba_bytes=ATLAS_WIDTH*height*4)


def build_fonts(font_dir=None):
  font_dir = Path(font_dir) if font_dir is not None else Path(__file__).resolve().parent
  codepoints = required_codepoints(font_dir)
  results = [_atlas(font_dir, weight, codepoints) for weight in ("Regular", "Bold")]
  print(json.dumps(results))
  return results


if __name__ == "__main__":
  build_fonts()
