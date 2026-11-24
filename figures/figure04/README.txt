Figure 4 source reconstruction

Run plot_figure4.py to draw the paper figure from records/a.npz through records/e.npz. Each signal is checked against its SHA-256 in records/sources.csv. Run prepare_records.py to reconstruct the plotting records from the included inputs, as described in ../README.md.

The mine examples retain vertical counts before the independent display normalization for each panel. The external landslide segment retains the source amplitude scale. Its original physical unit, station, UTC and component orientation are unspecified. The script writes the five panels used in the manuscript.

The source map and ranking are in sources/source_release_map.csv, sources/mat_provenance.json and records/slide_ranking.csv. Mine continuous traces are filtered before window extraction. For the external example the available 10 s MAT segment is filtered directly, since there are no samples outside that window. The paper uses one individual segment. Matching scores are used for choosing a displayed example and are not classification performance measures.
