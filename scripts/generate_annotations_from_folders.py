#!/usr/bin/env python3
"""Generate a simple annotations.json (list of {filename, species, count})
by scanning species-named subfolders.

Usage examples:
python scripts/generate_annotations_from_folders.py \
    --input_dirs /kaggle/input/pip-ceylonicusbat-species /kaggle/input/pip-tenuisbat-species \
    --output /kaggle/working/data/annotations.json

Or to scan the repo-local `data/` folder and write `data/annotations.json`:
python scripts/generate_annotations_from_folders.py --input_dirs data --output data/annotations.json
"""
import argparse
from pathlib import Path
import json


def generate(input_dirs, output_path, default_count=1, extensions=None):
    if extensions is None:
        extensions = {'.wav', '.WAV', '.flac', '.FLAC'}

    entries = []
    seen = set()

    for d in input_dirs:
        p = Path(d)
        if not p.exists():
            print(f"Warning: input dir {p} does not exist, skipping")
            continue

        # If the input path points directly to a dataset folder that contains species subfolders,
        # iterate subfolders; otherwise, treat the path itself as a species folder if it contains wavs.
        subdirs = [x for x in p.iterdir() if x.is_dir()]
        if subdirs:
            species_dirs = subdirs
        else:
            species_dirs = [p]

        for sdir in species_dirs:
            species_name = sdir.name
            for audio in sdir.rglob('*'):
                if audio.suffix in extensions and audio.is_file():
                    fname = audio.name
                    # Avoid duplicates across multiple input dirs
                    key = (fname, species_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    entries.append({
                        'filename': fname,
                        'species': species_name,
                        'count': default_count
                    })

    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} annotations to {outp}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='One or more directories to scan. Each species should be in its own subfolder or the dir itself can be the species folder.')
    parser.add_argument('--output', default='data/annotations.json', help='Output annotations JSON path')
    parser.add_argument('--count', type=int, default=1, help='Default call count to write for each file')
    args = parser.parse_args()

    generate(args.input_dirs, args.output, default_count=args.count)


if __name__ == '__main__':
    main()
