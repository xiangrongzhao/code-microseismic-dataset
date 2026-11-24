"""Locate a catalogue table, including in earlier named releases."""

PREVIOUS_NAMES = {
    "five_class_event_catalog.csv": "event_catalog.csv",
    "waveform_to_event_id.csv": "event_records.csv",
    "blasting_station_count_distribution.csv": "blasting_station_distribution.csv"
}


def table_path(folder, name):
    current = folder / name
    if current.is_file():
        return current
    previous = folder / PREVIOUS_NAMES[name]
    return previous if previous.is_file() else current
