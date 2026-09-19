"""Documentation screenshots of the desktop shell from the browser demo.

Renders the four demo stages in light and dark appearance with headless
Chrome against a running Vite dev server (``npm run dev`` in apps/desktop),
then frames each capture as a macOS window (rounded corners, traffic lights,
shadow) for the README:

    .venv/bin/python apps/desktop/scripts/screenshots.py [http://localhost:1420]

The demo shows clearly labelled interface values only; nothing is measured.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "assets" / "branding"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH, HEIGHT = 1280, 840
STAGES = {"frames": 4000, "review": 5000, "run": 4300, "result": 16000}


def capture(url: str, stage: str, dark: bool, target: Path) -> None:
    flags = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
             f"--window-size={WIDTH},{HEIGHT}", f"--virtual-time-budget={STAGES[stage]}", f"--screenshot={target}"]
    if dark:
        flags += ["--force-dark-mode", "--enable-features=WebContentsForceDark:inversion_method/cielab_based,image_behavior/none"]
    flags.append(f"{url}/?demo={stage}&mac=1")
    subprocess.run(flags, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def frame(shot: Image.Image, dark: bool) -> Image.Image:
    scale = 2
    radius = 12 * scale
    pad = 48 * scale
    width, height = shot.size
    canvas = Image.new("RGBA", (width + 2 * pad, height + 2 * pad), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((pad, pad + 10 * scale, pad + width, pad + height + 10 * scale), radius=radius, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18 * scale))
    canvas = Image.alpha_composite(canvas, shadow)
    mask = Image.new("L", shot.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    window = Image.new("RGBA", shot.size, (0, 0, 0, 0))
    window.paste(shot.convert("RGBA"), (0, 0), mask)
    draw = ImageDraw.Draw(window)
    # Traffic lights over the toolbar's reserved area.
    for index, colour in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
        cx, cy, r = (20 + index * 20) * scale, 26 * scale, 6 * scale
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=colour)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, outline=(255, 255, 255, 60) if dark else (0, 0, 0, 40), width=scale)
    canvas.alpha_composite(window, (pad, pad))
    return canvas


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1420"
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for stage in STAGES:
            for dark in (False, True):
                raw = Path(tmp) / f"{stage}-{'dark' if dark else 'light'}.png"
                capture(url, stage, dark, raw)
                shot = Image.open(raw)
                framed = frame(shot, dark)
                target = OUT / f"ultra-fast-wbpp-{stage}-{'dark' if dark else 'light'}.png"
                framed.save(target, "PNG", optimize=True)
                print(target.relative_to(ROOT), framed.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
