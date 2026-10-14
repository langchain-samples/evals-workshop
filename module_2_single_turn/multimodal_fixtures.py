"""Module 2 — synthetic receipts for the multimodal eval.

Every image in the expense-receipts dataset is drawn here, in code, from a small
spec. Two reasons:

  1. Nothing to download and nothing sensitive: no real receipts, no stock photos.
  2. The answer key comes from the same spec that drew the pixels, so the expected
     total can't drift from what the image shows.

A receipt can be drawn ``degraded`` (rotated, blurred, low contrast, downscaled) to
stand in for a phone photo. Those examples are tagged ``image_quality: degraded`` so
you can slice results by it.

Run:  python module_2_single_turn/multimodal_fixtures.py   # writes PNGs/CSVs to look at
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WIDTH = 520
MARGIN = 28


@dataclass(frozen=True)
class Receipt:
    vendor: str
    date: str  # YYYY-MM-DD, printed on the receipt
    lines: tuple[tuple[str, float], ...]
    tax: float = 0.0
    address: str = "100 Main St, Springfield"
    degraded: bool = False

    @property
    def total(self) -> float:
        return round(sum(amount for _, amount in self.lines) + self.tax, 2)


@dataclass(frozen=True)
class Statement:
    """A card statement export: plain CSV, the non-image attachment."""

    rows: tuple[tuple[str, str, float], ...]  # (date, description, amount)
    columns: tuple[str, ...] = field(default=("date", "description", "amount"))

    def to_csv(self) -> bytes:
        out = [",".join(self.columns)]
        out += [f"{d},{desc},{amount:.2f}" for d, desc, amount in self.rows]
        return ("\n".join(out) + "\n").encode()


def _font(size: int):
    from PIL import ImageFont

    # Pillow >= 10.1 ships a scalable default font, so no font files are needed.
    return ImageFont.load_default(size=size)


def receipt_png(receipt: Receipt) -> bytes:
    """Draw a receipt as a PNG and return the bytes."""
    from PIL import Image, ImageDraw

    rows = len(receipt.lines) + (1 if receipt.tax else 0)
    height = 260 + 40 * rows
    img = Image.new("RGB", (WIDTH, height), "white")
    draw = ImageDraw.Draw(img)
    big, body = _font(34), _font(24)

    y = MARGIN
    draw.text((WIDTH // 2, y), receipt.vendor.upper(), font=big, fill="black", anchor="mt")
    y += 48
    draw.text((WIDTH // 2, y), receipt.address, font=body, fill="#444", anchor="mt")
    y += 36
    draw.text((MARGIN, y), f"Date: {receipt.date}", font=body, fill="black")
    y += 40
    draw.line((MARGIN, y, WIDTH - MARGIN, y), fill="black", width=2)
    y += 16

    def row(label: str, amount: float, font=body) -> None:
        nonlocal y
        draw.text((MARGIN, y), label, font=font, fill="black")
        draw.text((WIDTH - MARGIN, y), f"${amount:,.2f}", font=font, fill="black", anchor="ra")
        y += 40

    for label, amount in receipt.lines:
        row(label, amount)
    if receipt.tax:
        row("Tax", receipt.tax)
    draw.line((MARGIN, y, WIDTH - MARGIN, y), fill="black", width=2)
    y += 16
    row("TOTAL", receipt.total, font=_font(30))

    if receipt.degraded:
        img = _degrade(img)
    return _png(img)


def not_a_receipt_png() -> bytes:
    """An image that is NOT a receipt — a welcome banner attached by mistake.

    The right behaviour is to ignore it. A model that "reads" a total off this is
    hallucinating, and the receipt-count evaluator catches it.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (WIDTH, 300), "#2f6fde")
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 60, 200, 220), fill="#ffd34d")
    draw.text((230, 110), "WELCOME", font=_font(48), fill="white")
    draw.text((230, 170), "to the team!", font=_font(30), fill="white")
    return _png(img)


def _degrade(img):
    """Make a clean render look like a so-so phone photo."""
    from PIL import ImageEnhance, ImageFilter

    img = ImageEnhance.Contrast(img).enhance(0.55)
    img = img.rotate(6, expand=True, fillcolor="#d9d4c7")
    img = img.filter(ImageFilter.GaussianBlur(0.9))
    return img.resize((img.width * 3 // 4, img.height * 3 // 4))


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    # Write every attachment in the dataset to a gitignored folder so you can see
    # exactly what the model (and the judge) will be shown.
    from module_2_single_turn.multimodal_datasets import EXAMPLES, build_attachments

    out_dir = Path(__file__).resolve().parent.parent / "fixtures" / "receipts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = {"image/png": "png", "text/csv": "csv"}
    for example in EXAMPLES:
        for name, (mime, data) in build_attachments(example).items():
            path = out_dir / f"{example['id']}__{name}.{ext[mime]}"
            path.write_bytes(data)
            print(f"wrote {path.relative_to(out_dir.parent.parent)}")
