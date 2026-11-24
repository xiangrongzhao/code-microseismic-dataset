"""Regenerate the numeric panels and use the supplied authored artwork."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTWORK = {
    1: ["fig_1.pdf"],
    2: ["fig2-a.pdf", "fig2-b.pdf"],
    7: ["fig7-a.pdf", "fig7-b.pdf", "fig7-c.pdf"],
}


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


def reproduce(number, out):
    out.mkdir(parents=True, exist_ok=True)
    for name in ARTWORK.get(number, []):
        src = HERE / "reference" / name
        if src.exists():
            shutil.copyfile(src, out / name)
    if number == 3:
        run(HERE / "figure03/plot_coverage.py", out / "fig3.pdf")
    elif number == 4:
        run(HERE / "figure04/plot_figure4.py", "--out", out)
    elif number == 6:
        for name in ("plot_raw_components.py", "plot_classified_window.py"):
            run(HERE / "figure06" / name, "--out", out)
    elif number == 7:
        for name in ("plot_fig7d_feature_space.py", "plot_fig7e_baseline_sanity.py"):
            run(HERE / "figure07" / name, "--out", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", *map(str, [1, 2, 3, 4, 6, 7])], default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    dest = args.out.resolve()
    if dest == ROOT or (ROOT in dest.parents and (ROOT / "outputs") not in [dest, *dest.parents]):
        parser.error("Use outputs/ inside the repository, or a new directory outside the repository.")
    numbers = [1, 2, 3, 4, 6, 7] if args.figure == "all" else [int(args.figure)]
    for n in numbers:
        print(f"Figure {n} -> {dest / ('figure%02d' % n)}", flush=True)
        reproduce(n, dest / ("figure%02d" % n))
