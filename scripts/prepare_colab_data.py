import zipfile
import time
from pathlib import Path

test_dir = Path("dataset_official/test")
zip_out = Path("test_data.zip")

print("Compressing official test dataset into test_data.zip for Colab upload...")
t0 = time.time()
with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for f in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
        fpath = test_dir / f
        if fpath.exists():
            print(f"  Adding {f} ({fpath.stat().st_size / 1e6:.1f} MB)...")
            z.write(fpath, arcname=f)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s! test_data.zip size: {zip_out.stat().st_size / 1e6:.1f} MB")
