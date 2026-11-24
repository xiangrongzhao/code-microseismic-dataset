"""Export the visible submitted diagrams and composite panels as SVG."""
import argparse
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
PANELS = ["fig_1", "fig2-a", "fig2-b", "fig7-a", "fig7-b", "fig7-c"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in PANELS:
        with fitz.open(str(HERE / "reference" / (name + ".pdf"))) as original:
            page = original[0]
            svg = page.get_svg_image(text_as_path=False)
            (args.output / (name + ".svg")).write_text(svg, encoding="utf-8")
