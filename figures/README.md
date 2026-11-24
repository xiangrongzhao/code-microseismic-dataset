# Figure scripts and source data

[Final figure gallery](manuscript_figures.pdf) · [Submitted PDF panels](reference) · [PowerPoint material](powerpoint)

From the repository root, run `python figures/reproduce.py --figure 6` or `python figures/reproduce.py --figure all`. Numerical outputs are written to `outputs/figures/figureNN`. Submitted PDFs in `reference/` keep the author-selected crop boxes. LaTeX composes the final figures from those PDFs. To compile the gallery, run `pdflatex manuscript_layout.tex` in this directory.

| Figure | Script or editable source | Plot inputs and scope |
|---|---|---|
| 1. Construction workflow | [PowerPoint flowchart](powerpoint/figure01_workflow.pptx) | Text and standard shapes. See [workflows](../docs/workflows.md). |
| 2. Network layout | Authored map / field / interface panels in `reference/` | Arranged in WPS Presentation; not regenerated from waveforms. |
| 3. Recording availability | [plot_coverage.py](figure03/plot_coverage.py), [export_workbook.py](figure03/export_workbook.py) | From [data_20250601_20250720.xls](../data/data%20logs/data_20250601_20250720.xls). |
| 4. Five waveform examples | [plot_figure4.py](figure04/plot_figure4.py) | Five class examples (`microseismic`, `blasting`, `small_landslide`, `mechanical_mining`, `transport_vehicles`). Panel (c) uses an external landslide segment. |
| 6. Released waveform examples | [plot_raw_components.py](figure06/plot_raw_components.py), [plot_classified_window.py](figure06/plot_classified_window.py) | Bundled miniSEED and blasting NPY under [figure06/input](figure06/input). |
| 7. Technical validation | [plot_validation.py](../06_validation/plot_validation.py), [figure07 scripts](figure07) | [validation_results/four_class](../validation_results/four_class); PowerPoint count/split charts under [powerpoint](powerpoint). |

## Plot inputs and reruns

- Figure 3: `python figures/figure03/export_workbook.py outputs/availability.csv --check-pdf`
- Figure 4: `python figures/figure04/prepare_records.py --out outputs/figure04_records` then `python figures/figure04/plot_figure4.py --records outputs/figure04_records --out outputs/figure04_from_raw`
- Figure 7(d): `python 06_validation/feature_statistics.py --results validation_results/four_class --out outputs/feature_statistics`

For vector inspection of authored panels, install `figures/requirements-source.txt` and run `python figures/export_artwork.py outputs/artwork`.
