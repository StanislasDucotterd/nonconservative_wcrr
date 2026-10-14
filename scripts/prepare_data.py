#!/usr/bin/env python
"""Build the HDF5 image files read by ncr/data.py from a folder of images: one grayscale image per key ("0", "1", ...),
float32 in [0, 1], shape (1, H, W).
    python scripts/prepare_data.py /path/to/BSD400 data/bsd400.h5     # the 400 training images (BSD400 / BSD432 split)
    python scripts/prepare_data.py /path/to/BSD68  data/bsd68.h5      # the 68 test images
Images are read in sorted file-name order; this order fixes the validation noise seeds, so a different order changes the
per-image PSNRs slightly (not the means)."""
import argparse
import os
import h5py
import numpy as np
from PIL import Image

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("out")
    args = ap.parse_args()
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(EXT))
    if not files:
        raise SystemExit(f"no images found in {args.folder}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with h5py.File(args.out, "w") as h5:
        for i, f in enumerate(files):
            img = np.asarray(Image.open(os.path.join(args.folder, f)).convert("L"), dtype=np.float32) / 255.0
            h5.create_dataset(str(i), data=img[None], compression="gzip")
    print(f"{len(files)} images written to {args.out}")


if __name__ == "__main__":
    main()
