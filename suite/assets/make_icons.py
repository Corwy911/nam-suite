"""Draws the three icons of NAM Suite (tube = trainer, dial = capture, both = suite) with Pillow, supersampled, as multi-size .ico.
Style: the gunmetal / champagne / amber of the two apps' skin. Run: python assets/make_icons.py"""
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
S = 1024   # drawing size; scaled down for the sizes of the .ico


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def vgrad(size, top, bottom):
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    for y in range(size[1]):
        d.line([(0, y), (size[0], y)], fill=lerp(top, bottom, y / max(1, size[1] - 1)))
    return img


def plate():
    """Rounded gunmetal square with a champagne edge."""
    m = Image.new("L", (S, S), 0)
    ImageDraw.Draw(m).rounded_rectangle((24, 24, S - 24, S - 24), radius=190, fill=255)
    base = vgrad((S, S), (92, 94, 98), (34, 36, 40))
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(base, (0, 0), m)
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((24, 24, S - 24, S - 24), radius=190, outline=(214, 186, 120, 255), width=16)
    d.rounded_rectangle((52, 52, S - 52, S - 52), radius=166, outline=(0, 0, 0, 90), width=5)
    return out


def glow(img, center, radius, color, strength=1.0):
    g = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(g).ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=color + (int(200 * strength),))
    g = g.filter(ImageFilter.GaussianBlur(radius * .55))
    return Image.alpha_composite(img, g)


def tube(img, cx, top, bottom, width, lit=True):
    d = ImageDraw.Draw(img)
    l, r = cx - width // 2, cx + width // 2
    glass = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass)
    gd.rounded_rectangle((l, top, r, bottom), radius=width // 2, fill=(20, 22, 26, 235), outline=(200, 205, 210, 150), width=8)
    img = Image.alpha_composite(img, glass)
    if lit:
        img = glow(img, (cx, (top + bottom) // 2 + 40), width, (255, 150, 40), 1.0)
    d = ImageDraw.Draw(img)
    # plates and the two glowing filaments
    d.rounded_rectangle((l + 34, top + 120, r - 34, bottom - 70), radius=10, outline=(170, 170, 165, 255), width=10)
    for fx in (cx - 26, cx + 26):
        d.rounded_rectangle((fx - 9, top + 136, fx + 9, bottom - 88), radius=9, fill=(255, 196, 96, 255) if lit else (90, 50, 26, 255))
    # getter cap and highlight
    d.ellipse((l + 28, top + 40, r - 28, top + 120), fill=(190, 196, 200, 255))
    d.arc((l + 14, top + 30, l + 90, top + 230), 200, 290, fill=(255, 255, 255, 190), width=12)
    # base
    d.rounded_rectangle((l + 6, bottom - 4, r - 6, bottom + 60), radius=14, fill=(58, 38, 22, 255))
    for px in (cx - 52, cx - 18, cx + 18, cx + 52):
        d.rectangle((px - 6, bottom + 60, px + 6, bottom + 100), fill=(190, 190, 184, 255))
    return img


def icon_trainer():
    img = plate()
    img = glow(img, (S // 2, S // 2 + 60), 330, (255, 120, 30), .55)
    return tube(img, S // 2, 170, 720, 330)


def icon_capture():
    img = plate()
    d = ImageDraw.Draw(img)
    cx, cy, R = S // 2, S // 2 + 10, 330
    # dial face
    face = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    fd = ImageDraw.Draw(face)
    fd.ellipse((cx - R, cy - R, cx + R, cy + R), fill=(236, 226, 196, 255), outline=(120, 92, 40, 255), width=18)
    img = Image.alpha_composite(img, face)
    d = ImageDraw.Draw(img)
    import math
    for i in range(0, 21):
        a = math.radians(-130 + i * 13)
        r1, r2 = R - 38, R - (90 if i % 5 == 0 else 62)
        d.line([(cx + r1 * math.sin(a), cy - r1 * math.cos(a)), (cx + r2 * math.sin(a), cy - r2 * math.cos(a))], fill=(40, 34, 24, 255), width=14 if i % 5 == 0 else 7)
    a = math.radians(38)
    d.line([(cx - 60 * math.sin(a), cy + 60 * math.cos(a)), (cx + (R - 70) * math.sin(a), cy - (R - 70) * math.cos(a))], fill=(200, 40, 16, 255), width=20)
    d.ellipse((cx - 38, cy - 38, cx + 38, cy + 38), fill=(190, 190, 184, 255), outline=(60, 60, 58, 255), width=8)
    return img


def icon_suite():
    img = plate()
    img = glow(img, (S // 2, S // 2 + 80), 360, (255, 120, 30), .4)
    img = tube(img, 330, 250, 720, 230)
    img = tube(img, 694, 250, 720, 230)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((120, 820, S - 120, 880), radius=26, fill=(214, 186, 120, 255))
    return img


def save(name, img):
    img = img.convert('RGBA'); img.putalpha(ImageChops.multiply(img.split()[3], plate().split()[3]))   # keep glows inside the plate
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]
    img.convert("RGBA").save(HERE / name, format="ICO", sizes=sizes)
    img.convert("RGBA").resize((256, 256), Image.LANCZOS).save(HERE / (Path(name).stem + ".png"))
    print("wrote", name)


save("nam-trainer.ico", icon_trainer())
save("nam-capture.ico", icon_capture())
save("nam-suite.ico", icon_suite())
