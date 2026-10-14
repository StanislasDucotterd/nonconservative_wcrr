#!/usr/bin/env python
"""Train a fixed-point denoiser.  Example:
    python train.py --config configs/csr_convex.yaml --device cuda:0
    python train.py --config configs/crr_spline_delta_last_c64_tv2e-6.yaml --set train.epochs=400 exp_name=final_quick
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncr.utils import load_config
from ncr.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=100 model.channels=[1,4,8,32]")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tr = Trainer(cfg, args.device)
    if args.resume:
        tr.resume(args.resume)
    tr.train()


if __name__ == "__main__":
    main()
