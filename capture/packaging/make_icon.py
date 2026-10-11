"""Draws the NAM Capture app icon: a knurled gold knob with a partial tick scale, sitting above a small engraved
nameplate, mounted on the mahogany cabinet edge - the "Kantata" 1980s-radio look adopted for the app's own UI
(see static/retro.css). Builds AppIcon.icns next to this script. Needs Pillow (already in the project venv) and
macOS's `iconutil`.

    venv/bin/python packaging/make_icon.py
"""

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
SIZE = 1024
FONT_PATH = "/System/Library/Fonts/Avenir Next Condensed.ttc"   # index 0 = Bold - the app's own --font-head

MAHOGANY_TOP = (109, 37, 19)     # #6d2513
MAHOGANY_MID = (74, 23, 11)      # #4a170b
MAHOGANY_BOT = (47, 14, 6)       # #2f0e06
CHAMPAGNE = (203, 178, 122)      # retro.css --champagne
GOLD_LIGHT = (244, 227, 164)     # retro.css knurl highlight
GOLD_DARK = (164, 134, 63)       # retro.css knurl shadow
GOLD_EDGE = (60, 40, 10)         # dark groove between ring and cap
CAP_FILL = (130, 130, 124)       # retro.css --knob-cap base
POINTER = (246, 244, 234)        # retro.css knob pointer
SCREW_LIGHT = (220, 219, 214)
SCREW_DARK = (111, 110, 106)
STRIP_TOP = (30, 30, 28)         # retro.css --strip1
STRIP_BOT = (12, 12, 11)         # retro.css --strip2
PLATE_TEXT = (239, 219, 155)     # retro.css --gold1, warmed


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def radial_blob(size: int, cx: float, cy: float, r: float, color, blur: float) -> Image.Image:
    """A soft round patch of `color` (RGBA), for shading built by compositing blurred blobs rather than a true
    radial gradient (Pillow has none built in)."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(layer).ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return layer.filter(ImageFilter.GaussianBlur(blur)) if blur else layer


def draw_tracked_text(draw: ImageDraw.ImageDraw, cx: float, cy: float, text: str, font: ImageFont.FreeTypeFont, fill, tracking: float) -> None:
    """Centers `text` on (cx, cy) with extra letter-spacing - Pillow has no built-in tracking, so each glyph is placed by hand."""
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = cx - total / 2
    for ch, w in zip(text, widths):
        draw.text((x, cy), ch, font=font, fill=fill, anchor="lm")
        x += w + tracking


def draw_icon() -> Image.Image:
    SS = 4                                    # supersample factor: smooth curves and clean pie-slice edges
    big = SIZE * SS
    canvas = Image.new("RGBA", (big, big), (0, 0, 0, 0))

    # the mahogany-lacquer cabinet edge (same three stops as the page background in retro.css)
    bg = Image.new("RGB", (big, big))
    px = bg.load()
    for y in range(big):
        t = y / big
        if t < 0.55:
            row = tuple(round(MAHOGANY_TOP[c] + (MAHOGANY_MID[c] - MAHOGANY_TOP[c]) * (t / 0.55)) for c in range(3))
        else:
            row = tuple(round(MAHOGANY_MID[c] + (MAHOGANY_BOT[c] - MAHOGANY_MID[c]) * ((t - 0.55) / 0.45)) for c in range(3))
        for x in range(big):
            px[x, y] = row
    bg = bg.convert("RGBA")
    bg.putalpha(rounded_mask(big, int(big * 0.225)))
    canvas.alpha_composite(bg)

    # a soft diagonal sheen across the lacquer, like the header dial's glass highlight - a bit of shine, not a texture
    sheen = Image.new("L", (big, big), 0)
    sd = ImageDraw.Draw(sheen)
    band = big * 0.55
    sd.polygon([(-band, 0), (0, 0), (big * 0.55, big), (big * 0.15, big)], fill=70)
    sheen = sheen.filter(ImageFilter.GaussianBlur(big * 0.05))
    sheen_layer = Image.new("RGBA", (big, big), (255, 255, 255, 255))
    sheen_layer.putalpha(ImageChops.multiply(sheen, bg.split()[3]))
    canvas.alpha_composite(sheen_layer)

    d = ImageDraw.Draw(canvas)

    # a champagne inset edge, like every metal panel in the app
    inset = big * 0.03
    d.rounded_rectangle([inset, inset, big - inset, big - inset], radius=int(big * 0.195),
                         outline=CHAMPAGNE + (150,), width=max(2, int(big * 0.0055)))

    # corner screws, like .card::before/::after
    screw_r = big * 0.021
    for sx in (big * 0.115, big * 0.885):
        canvas.alpha_composite(radial_blob(big, sx, big * 0.115, screw_r * 1.6, SCREW_DARK + (255,), big * 0.006))
        d.ellipse([sx - screw_r, big * 0.115 - screw_r, sx + screw_r, big * 0.115 + screw_r], fill=SCREW_LIGHT)
        d.ellipse([sx - screw_r * 0.55, big * 0.115 - screw_r * 0.55, sx + screw_r * 0.55, big * 0.115 + screw_r * 0.55], fill=SCREW_DARK)

    cx, cy = big * 0.5, big * 0.44                            # the knob sits a bit above centre, to leave room for the nameplate

    # soft drop shadow under the whole knob
    canvas.alpha_composite(radial_blob(big, cx, cy + big * 0.02, big * 0.30, (0, 0, 0, 190), big * 0.022))

    # a partial tick scale around the knob (like .knob::before's own dial ticks): 11 marks over 270 degrees, the
    # open gap left at the bottom for the nameplate below - an instrument scale, not just a bare knob
    ring_r = big * 0.30
    tick_in, tick_out = ring_r * 1.14, ring_r * 1.30
    for i in range(11):
        ang = math.radians(-135 + i * 27 - 90)                # -90: 0 degrees in PIL's pieslice frame is 3 o'clock, so shift to 12
        x1, y1 = cx + tick_in * math.cos(ang), cy + tick_in * math.sin(ang)
        x2, y2 = cx + tick_out * math.cos(ang), cy + tick_out * math.sin(ang)
        d.line([x1, y1, x2, y2], fill=CHAMPAGNE + (210,), width=max(2, int(big * 0.008)))

    # the knurled gold ring: alternating light/dark wedges every 4 degrees (retro.css's repeating-conic-gradient)
    ring_box = [cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r]
    step = 4
    for angle in range(0, 360, step * 2):
        d.pieslice(ring_box, angle - 90, angle - 90 + step, fill=GOLD_LIGHT)
        d.pieslice(ring_box, angle - 90 + step, angle - 90 + step * 2, fill=GOLD_DARK)
    # a thin dark rim so the ring reads as a distinct piece against the cabinet
    d.ellipse(ring_box, outline=GOLD_EDGE, width=max(2, int(big * 0.006)))

    # the dark groove and the dished cap it sits in
    groove_r = ring_r * 0.86
    d.ellipse([cx - groove_r, cy - groove_r, cx + groove_r, cy + groove_r], fill=GOLD_EDGE)
    cap_r = ring_r * 0.80
    d.ellipse([cx - cap_r, cy - cap_r, cx + cap_r, cy + cap_r], fill=CAP_FILL)

    # the dish: a thin shadow along the top-left rim and a thin glint along the bottom-right rim, like .knob::after's
    # inset box-shadow - built as crescents (the cap circle minus itself shifted towards the light), not center blobs
    cap_mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(cap_mask).ellipse([cx - cap_r, cy - cap_r, cx + cap_r, cy + cap_r], fill=255)
    shift = cap_r * 0.12

    def rim_crescent(dx: float, dy: float) -> Image.Image:
        shifted = Image.new("L", (big, big), 0)
        ImageDraw.Draw(shifted).ellipse([cx - cap_r + dx, cy - cap_r + dy, cx + cap_r + dx, cy + cap_r + dy], fill=255)
        return ImageChops.subtract(cap_mask, shifted).filter(ImageFilter.GaussianBlur(big * 0.01))

    dish = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    shadow_a = rim_crescent(shift, shift).point(lambda v: v * 0.65)
    dish.paste((0, 0, 0, 255), (0, 0), shadow_a)
    glint_a = rim_crescent(-shift, -shift).point(lambda v: v * 0.35)
    glint = Image.new("RGBA", (big, big), (255, 255, 255, 255))
    glint.putalpha(glint_a)
    dish.alpha_composite(glint)
    canvas.alpha_composite(dish)

    # the pointer, straight up (a clean, legible rest position for an icon)
    pw = cap_r * 0.11
    d.rounded_rectangle([cx - pw, cy - cap_r * 0.86, cx + pw, cy - cap_r * 0.22], radius=pw, fill=POINTER)

    # the nameplate: a recessed black strip, printed in gold caps, like every field label in the app
    plate_w, plate_h = big * 0.72, big * 0.105
    plate_y = big * 0.855
    plate_box = [cx - plate_w / 2, plate_y - plate_h / 2, cx + plate_w / 2, plate_y + plate_h / 2]
    strip = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    sd2 = ImageDraw.Draw(strip)
    radius = int(plate_h * 0.3)
    for i in range(int(plate_h)):
        yy = plate_box[1] + i
        t = i / plate_h
        row = tuple(round(STRIP_TOP[c] + (STRIP_BOT[c] - STRIP_TOP[c]) * t) for c in range(3))
        sd2.line([plate_box[0], yy, plate_box[2], yy], fill=row + (255,))
    plate_mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(plate_mask).rounded_rectangle(plate_box, radius=radius, fill=255)
    strip.putalpha(ImageChops.multiply(strip.split()[3], plate_mask))
    canvas.alpha_composite(strip)
    d.rounded_rectangle(plate_box, radius=radius, outline=(0, 0, 0, 140), width=max(2, int(big * 0.004)))

    font = ImageFont.truetype(FONT_PATH, int(plate_h * 0.52), index=0)
    draw_tracked_text(d, cx, plate_y - plate_h * 0.015, "NAM CAPTURE", font, PLATE_TEXT, tracking=plate_h * 0.16)

    return canvas.resize((SIZE, SIZE), Image.LANCZOS)


def build_iconset(img: Image.Image) -> Path:
    iconset = HERE / "AppIcon.iconset"
    iconset.mkdir(exist_ok=True)
    for base, px in ((16, 16), (16, 32), (32, 32), (32, 64), (128, 128), (128, 256), (256, 256), (256, 512), (512, 512), (512, 1024)):
        suffix = "" if px == base else "@2x"
        name = f"icon_{base}x{base}{suffix}.png"
        img.resize((px, px), Image.LANCZOS).save(iconset / name)
    return iconset


def main() -> None:
    img = draw_icon()
    img.save(HERE / "icon_preview.png")
    iconset = build_iconset(img)
    out = HERE / "AppIcon.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
    print("written:", out)


if __name__ == "__main__":
    main()
