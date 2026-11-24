"""Export inclusive June/July station availability ranges and optionally check all cells against the reference PDF."""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import xlrd

HERE = Path(__file__).resolve().parent


def export(workbook):
    sheet = xlrd.open_workbook(str(workbook)).sheet_by_index(0)
    if sheet.row_values(0)[1:3] != ["Jun", "Jul"]:
        raise ValueError("Expected the station log with Jun and Jul columns.")
    dates = [date(2025, 6, 1) + timedelta(days=i) for i in range(50)]
    matrix = {}
    for i in range(1, sheet.nrows):
        station, june, july = sheet.row_values(i)[:3]
        station = str(int(station))
        if station in matrix:
            raise ValueError(f"Duplicate station {station}")
        available = set()
        for month, values in [(6, june), (7, july)]:
            if isinstance(values, float):
                values = str(int(values))
            for token in str(values).split():
                ends = [int(text[-2:]) if len(text) == 8 else int(text) for text in token.split("-")]
                available.update(date(2025, month, day) for day in range(ends[0], ends[-1] + 1))
        matrix[station] = [int(day in available) for day in dates]
    return dates, matrix


def compare_pdf(pdf, matrix):
    import fitz
    with fitz.open(pdf) as doc:
        page = doc[0]
        paths = page.get_drawings()
        labels = [(w[4], (w[1]+w[3])/2) for w in page.get_text("words") if w[4] in matrix]
        gray = [p["rect"] for p in paths if p["fill"] and .85 < p["fill"][0] < .88 and p["rect"].width > 800]
        green = [p["rect"] for p in paths if p["fill"] and .16 < p["fill"][0] < .18]
        if len(gray) != len(matrix) or {s for s, _ in labels} != set(matrix):
            raise ValueError("Reference station labels or row extents do not match.")
        for station, label_y in labels:
            row = min(gray, key=lambda r: abs((r.y0+r.y1)/2-label_y))
            y = (row.y0+row.y1)/2
            for j, expected in enumerate(matrix[station]):
                point = fitz.Point(row.x0 + row.width*(j+.5)/50, y)
                if any(g.contains(point) for g in green) != bool(expected):
                    raise ValueError(f"Mismatch for station {station}, date column {j+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workbook", type=Path, default=HERE.parents[1] / "data" / "data logs" / "data_20250601_20250720.xls")
    parser.add_argument("--check-pdf", action="store_true")
    args = parser.parse_args()
    dates, values = export(args.workbook)
    if args.check_pdf:
        compare_pdf(str(HERE.parent / "reference" / "fig3.pdf"), values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf8", newline="") as f:
        out = csv.writer(f)
        out.writerow(["station_id", *map(str, dates)])
        out.writerows([[s, *values[s]] for s in sorted(values)])
    print(f"{len(values)} stations, {len(dates)} dates")
