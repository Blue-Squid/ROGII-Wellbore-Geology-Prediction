import glob
import polars as pl

files = glob.glob('data/raw/train/*.csv')
all_columns = set()

for f in files:
    # scan_csv only reads the header, making this instant
    schema_len = len(pl.scan_csv(f).schema)
    if schema_len == 13:
        all_columns.update(pl.scan_csv(f).schema.keys())

common_columns = all_columns.intersection(set(pl.scan_csv(files[0]).schema.keys()))
print(common_columns)