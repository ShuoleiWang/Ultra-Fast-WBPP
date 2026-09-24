"""Render the Ultra-Fast WBPP application icon and brand mark.

The mark is a macOS-style rounded square: a deep-sky gradient, a faint star
field, three translucent light frames converging on a single bright star with
diffraction spikes (many frames, one verified master).  Everything is drawn
with Pillow and NumPy so the icon set is reproducible from this file.

    .venv/bin/python apps/desktop/scripts/brand_icon.py

writes the Tauri icon set (PNG sizes, .icns via iconutil on macOS, .ico),
assets/branding/ultra-fast-wbpp-icon-1024.png and the SVG brand mark used in
the toolbar and the README.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
ICON_DIR = ROOT / "apps" / "desktop" / "src-tauri" / "icons"
BRANDING = ROOT / "assets" / "branding"
SIZE = 1024
# macOS icons leave a transparent margin; the artwork spans 824 px of 1024.
INSET = 100
RADIUS_FRACTION = 0.2237


STAR = (0.72, 0.36)


def gradient(size: int) -> Image.Image:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / (size - 1)
    top = np.array([7, 11, 40])
    mid = np.array([20, 38, 112])
    bottom = np.array([88, 44, 150])
    t = np.clip(y * 0.7 + x * 0.4, 0, 1)
    rgb = np.where((t < 0.55)[..., None], top + (mid - top) * (t / 0.55)[..., None], mid + (bottom - mid) * ((t - 0.55) / 0.45)[..., None])
    # Nebular haze: a teal veil low left, a magenta veil low right, a glow at the star.
    haze = np.exp(-(((x - 0.28) ** 2) / 0.06 + ((y - 0.74) ** 2) / 0.05))
    rgb = rgb + haze[..., None] * np.array([10, 60, 90])
    haze = np.exp(-(((x - 0.86) ** 2) / 0.05 + ((y - 0.88) ** 2) / 0.05))
    rgb = rgb + haze[..., None] * np.array([90, 30, 80])
    glow = np.exp(-(((x - STAR[0]) ** 2) / 0.07 + ((y - STAR[1]) ** 2) / 0.07))
    rgb = rgb + glow[..., None] * np.array([40, 70, 130])
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.full((size, size, 1), 255, dtype=np.uint8)
    return Image.fromarray(np.concatenate([rgb, alpha], axis=2), "RGBA")


def star_field(size: int, seed: int = 7) -> Image.Image:
    rng = np.random.default_rng(seed)
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _ in range(170):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.choice([1.2, 1.6, 2.2, 3.0], p=[0.45, 0.3, 0.18, 0.07])
        a = int(rng.uniform(70, 200))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(235, 240, 255, a))
    return layer.filter(ImageFilter.GaussianBlur(0.6))


def plates(size: int) -> Image.Image:
    """Three glass frames stepping up towards the star: many exposures, one master."""

    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    w, h = int(size * 0.34), int(size * 0.27)
    radius = int(size * 0.035)
    origin = (int(size * 0.15), int(size * 0.50))
    step = (int(size * 0.075), int(size * -0.075))
    stroke = max(3, size // 300)
    for index, alpha in enumerate((34, 60, 100)):
        x0 = origin[0] + step[0] * index
        y0 = origin[1] + step[1] * index
        # Fill with a vertical glass gradient (brighter at the top edge).
        fill = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        column = np.linspace(1.0, 0.45, h)[:, None]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 200; rgba[..., 1] = 218; rgba[..., 2] = 255
        rgba[..., 3] = (alpha * column * np.ones((1, w))).astype(np.uint8)
        fill = Image.fromarray(rgba, "RGBA")
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
        plate = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        plate.paste(fill, (x0, y0), mask)
        draw = ImageDraw.Draw(plate)
        draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=radius, outline=(236, 243, 255, min(255, alpha + 120)), width=stroke)
        # A few pinpoint stars inside each frame.
        rng = np.random.default_rng(11 + index)
        for _ in range(9):
            sx, sy = rng.uniform(x0 + 14, x0 + w - 14), rng.uniform(y0 + 14, y0 + h - 14)
            r = rng.uniform(1.6, 3.2)
            draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(255, 255, 255, int(rng.uniform(120, 230))))
        layer = Image.alpha_composite(layer, plate)
    return layer


def light_streak(size: int) -> Image.Image:
    """A soft beam from the frames to the star."""

    layer = Image.new("RGBA", (size * 2, size * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = size, size
    draw.ellipse((cx - size * 0.26, cy - size * 0.05, cx + size * 0.26, cy + size * 0.05), fill=(180, 210, 255, 110))
    layer = layer.filter(ImageFilter.GaussianBlur(size * 0.03)).rotate(-32, resample=Image.BICUBIC)
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.paste(layer, (int(size * 0.52) - size, int(size * 0.47) - size), layer)
    return result


def master_star(size: int) -> Image.Image:
    cx, cy = size * STAR[0], size * STAR[1]
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    for r, a in ((size * 0.20, 60), (size * 0.12, 110), (size * 0.06, 200)):
        gdraw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(170, 205, 255, a))
    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.035))
    layer = Image.alpha_composite(layer, glow)
    # Diffraction spikes as tapered blades: long vertical/horizontal, short diagonals.
    spikes = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(spikes)
    for angle, arm, base in ((0, size * 0.30, size * 0.016), (90, size * 0.30, size * 0.016), (45, size * 0.15, size * 0.010), (135, size * 0.15, size * 0.010)):
        dx, dy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        nx, ny = -dy, dx
        for direction in (1, -1):
            tip = (cx + dx * arm * direction, cy + dy * arm * direction)
            sdraw.polygon([(cx + nx * base, cy + ny * base), tip, (cx - nx * base, cy - ny * base)], fill=(244, 248, 255, 235))
    spikes = spikes.filter(ImageFilter.GaussianBlur(size * 0.0035))
    layer = Image.alpha_composite(layer, spikes)
    core = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(core)
    r = size * 0.032
    cdraw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 255))
    core = core.filter(ImageFilter.GaussianBlur(size * 0.008))
    return Image.alpha_composite(layer, core)


def grid_arcs(size: int) -> Image.Image:
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = size * STAR[0], size * STAR[1]
    width = max(1, size // 512)
    for r, a in ((size * 0.26, 70), (size * 0.42, 46), (size * 0.60, 28)):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(170, 200, 255, a), width=width)
    return layer


def rounded_mask(size: int, inset: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    side = size - 2 * inset
    draw.rounded_rectangle((inset, inset, inset + side, inset + side), radius=int(side * RADIUS_FRACTION), fill=255)
    return mask


def render_icon(size: int = SIZE, inset: int = INSET) -> Image.Image:
    art = gradient(size)
    for layer in (star_field(size), grid_arcs(size), light_streak(size), plates(size), master_star(size)):
        art = Image.alpha_composite(art, layer)
    # Edge highlight and a faint vignette keep the square from looking flat.
    vignette = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vignette)
    side = size - 2 * inset
    vdraw.rounded_rectangle((inset, inset, inset + side, inset + side), radius=int(side * RADIUS_FRACTION), outline=(255, 255, 255, 70), width=max(2, size // 400))
    art = Image.alpha_composite(art, vignette)
    icon = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    icon.paste(art, (0, 0), rounded_mask(size, inset))
    return icon


def write_png(image: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((size, size), Image.LANCZOS).save(path, "PNG", optimize=True)


def write_icns(image: Image.Image, path: Path) -> bool:
    if shutil.which("iconutil") is None:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for base in (16, 32, 128, 256, 512):
            write_png(image, iconset / f"icon_{base}x{base}.png", base)
            write_png(image, iconset / f"icon_{base}x{base}@2x.png", base * 2)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(path)], check=True)
    return True


BRAND_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" role="img" aria-label="Ultra-Fast WBPP">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0B1030"/><stop offset=".55" stop-color="#1A2C70"/><stop offset="1" stop-color="#4A2E8A"/>
    </linearGradient>
    <radialGradient id="glow" cx=".7" cy=".44" r=".42">
      <stop offset="0" stop-color="#9CC2FF" stop-opacity=".55"/><stop offset="1" stop-color="#9CC2FF" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#bg)"/>
  <rect x="2" y="2" width="60" height="60" rx="14" fill="url(#glow)"/>
  <rect x="2.5" y="2.5" width="59" height="59" rx="13.5" fill="none" stroke="#fff" stroke-opacity=".18"/>
  <g fill="#BED2FF" stroke="#E6F0FF">
    <rect x="12" y="27" width="24" height="20" rx="3" fill-opacity=".14" stroke-opacity=".45"/>
    <rect x="17" y="22" width="24" height="20" rx="3" fill-opacity=".22" stroke-opacity=".6"/>
    <rect x="22" y="17" width="24" height="20" rx="3" fill-opacity=".34" stroke-opacity=".8"/>
  </g>
  <g stroke="#F4F8FF" stroke-linecap="round">
    <path d="M45 15v26M32 28h26" stroke-width="1.4" stroke-opacity=".9"/>
    <path d="M38.5 21.5l13 13M51.5 21.5l-13 13" stroke-width="1" stroke-opacity=".6"/>
  </g>
  <circle cx="45" cy="28" r="3.2" fill="#fff"/>
  <circle cx="45" cy="28" r="6" fill="#fff" fill-opacity=".22"/>
</svg>
"""


def main() -> int:
    icon = render_icon()
    BRANDING.mkdir(parents=True, exist_ok=True)
    icon.save(BRANDING / "ultra-fast-wbpp-icon-1024.png", "PNG", optimize=True)
    (BRANDING / "ultra-fast-wbpp-mark.svg").write_text(BRAND_SVG)
    for size in (32, 64, 128):
        write_png(icon, ICON_DIR / f"{size}x{size}.png", size)
    write_png(icon, ICON_DIR / "128x128@2x.png", 256)
    for size in (30, 44, 71, 89, 107, 142, 150, 284, 310):
        write_png(icon, ICON_DIR / f"Square{size}x{size}Logo.png", size)
    write_png(icon, ICON_DIR / "StoreLogo.png", 50)
    write_png(icon, ICON_DIR / "icon.png", 512)
    icon.resize((256, 256), Image.LANCZOS).save(ICON_DIR / "icon.ico", format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    icns = write_icns(icon, ICON_DIR / "icon.icns")
    print(f"icon set written to {ICON_DIR} (icns: {'yes' if icns else 'skipped, no iconutil'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
