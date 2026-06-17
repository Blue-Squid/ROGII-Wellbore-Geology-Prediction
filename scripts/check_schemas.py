import glob
import polars as pl

files = glob.glob('data/raw/train/*.csv')
schemas = {}

for f in files:
    # scan_csv only reads the header, making this instant
    schema_len = len(pl.scan_csv(f).schema)
    schemas.setdefault(schema_len, []).append(f)

for length, paths in schemas.items():
    print(f'Columns: {length} -> Found in {len(paths)} files. Example: {paths[0]}')