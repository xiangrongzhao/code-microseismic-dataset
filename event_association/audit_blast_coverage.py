"""Check the two example-day blasting triggers against raw and retained data."""
import argparse
import csv
from pathlib import Path
from catalogue_paths import table_path

from obspy import UTCDateTime, read
from obspy.signal.trigger import classic_sta_lta, trigger_onset


def coverage(target, candidate_files):
    present, triggered = set(), set()
    for station, start, path in candidate_files:
        if not -5 <= target - start <= 310:
            continue
        for trace in read(str(path)):
            if trace.stats.channel != 'DHZ' or trace.stats.starttime > target-1 or trace.stats.endtime < target+1:
                continue
            present.add(station)
            trace.filter('bandpass', freqmin=1, freqmax=20, corners=4, zerophase=True)
            fs = trace.stats.sampling_rate
            characteristic = classic_sta_lta(trace.data, round(.5*fs), round(3*fs))
            for onset, end in trigger_onset(characteristic, 3.5, 1.5):
                if abs(trace.stats.starttime + onset/fs - target) <= 1.5:
                    triggered.add(station)
                    break
    return present, triggered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    args = p.parse_args()
    with (table_path(args.tables, "waveform_to_event_id.csv")).open(encoding='utf-8-sig') as f:
        blasts = [r for r in csv.DictReader(f) if r['record_label'] == 'blasting']
    raw_root = args.dataset/'raw waveform data'
    files = []
    for path in raw_root.glob('*.DHZ.miniseed'):
        station, date, clock, _, _ = path.name.split('.')
        files.append((station, UTCDateTime.strptime(date+clock, '%Y%m%d%H%M%S%f'), path))
    if not files:
        raise ValueError('The 1 June vertical raw example is required.')
    for record in blasts:
        if not record['trigger_on_utc'].startswith('2025-06-01'):
            continue
        target = UTCDateTime(record['trigger_on_utc'])
        present, triggered = coverage(target, files)
        retained = {r['station_id'] for r in blasts if abs(UTCDateTime(r['trigger_on_utc'])-target) <= 1.5}
        print(record['trigger_on_utc'], 'available / nearby triggers / retained', len(present), len(triggered), len(retained))
        print('Available:', sorted(present), 'Triggered:', sorted(triggered), 'Retained:', sorted(retained))


if __name__ == '__main__':
    main()
