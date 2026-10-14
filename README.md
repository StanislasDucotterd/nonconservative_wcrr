# Non-conservative ridge regularizers

Code of *From Critical Points to Fixed Points: Non-Conservative Ridge Regularizers for Image Reconstruction*
(L. A. Zavala-Mondragón and S. Ducotterd, 2026).

The model is the weakly convex ridge regularizer (WCRR) architecture with its transposed filter bank replaced by a second,
freely learned bank: the field

    M(x) = Vᵀ φ(W x),    V ≠ W,

is not the gradient of a regularizer, so a reconstruction is defined as a zero of Hᵀ(Hx − y) + λ M_σ(x), i.e. a fixed point
of the gradient iteration of CRR / WCRR with ∇R replaced by M. The fixed point is computed with SAGD (accelerated gradient
descent with adaptive restart, the CRR / WCRR solver) and the model is trained through the denoising fixed point x = y − M(x)
by implicit differentiation. The trained model (`trained_models/nonconservative_wcrr/last.pth`, 27k parameters) and the
numbers behind every table and figure of the paper (`results/`) are included.

## Installation

```bash
pip install -r requirements.txt          # torch >= 2.5, numpy, h5py, pyyaml, tqdm, matplotlib, pillow (+ tensorboard)
python tests/run_tests.py --device cuda:0
```

The sparse-view CT experiment additionally needs `odl`, `astra-toolbox` (both on conda-forge), `torchmetrics`, and a clone
of the [MFoE repository](https://github.com/StanislasDucotterd/MFoE), which provides the ODL / ASTRA operators, the
zeroth-order hyper-parameter search and the LoDoPaB-CT slices:

```bash
git clone https://github.com/StanislasDucotterd/MFoE
cd MFoE/inverse_problems/ct && tar xzf val_data.tar.gz && tar xzf test_data.tar.gz && python get_data.py && cd -
```

## Data

Training uses the 400 grayscale BSD images of CRR / WCRR (BSD400) and evaluation the 68 images of BSD68. Put the images of
each set in a folder and build the HDF5 files read by the code:

```bash
python scripts/prepare_data.py /path/to/BSD400 data/bsd400.h5
python scripts/prepare_data.py /path/to/BSD68  data/bsd68.h5
```

## Training

```bash
python train.py --config configs/nonconservative_wcrr.yaml --device cuda:0
```

`configs/nonconservative_wcrr.yaml` holds every setting of the paper's model: 64 filters of 13×13 (three 5×5 convolutions),
one antisymmetric linear spline shared by the channels with the WCRR noise-level scaling, σ uniform in [0, 30] during
training, L1 loss, Adam from 5e-3 to 5e-4 over 6000 steps of 64 crops of 64×64, SAGD forward (relative change 1e-4, at most
1000 iterations), Anderson acceleration on the adjoint system. The run writes `trained_models/nonconservative_wcrr/`
(`last.pth` is the model used everywhere; `train_log.csv` and `val_log.csv` of the paper's run are included). About two hours
on one GPU.

## Reproducing the paper

| item | command | output |
|---|---|---|
| Table 1, denoising on BSD68 | `python scripts/eval_bsd68.py --device cuda:0` | `results/bsd68/nonconservative_wcrr.json` (28.90 / 31.36 / 37.67 dB at σ = 25 / 15 / 5) |
| Table 2, sparse-view CT | `python scripts/ct_reconstruction.py --mfoe-dir /path/to/MFoE --angles 60 --lamb-init 4.09 --sigma-init 10` and the same with `--angles 40 --lamb-init 6.88 --sigma-init 7.06`, `--angles 20 --lamb-init 0.723 --sigma-init 36.7` | `results/ct/ours_<angles>/results.json` (35.23 / 0.837, 33.92 / 0.815, 31.32 / 0.770) |
| Fig. 1, convergence of SAGD on a CT slice | `python scripts/ct_residuals.py --mfoe-dir /path/to/MFoE --angles 60 --slice 0` then `python scripts/plot_ct_residuals.py results/ct/changes_60.json figures/convergence_ct.pdf 1500` | `figures/convergence_ct.pdf` |
| Figs. 2 and 3, filter pairs and spline | `python scripts/paper_figures.py` | `figures/filter_pairs.pdf`, `figures/spline.pdf` |
| Fig. 4, Jacobian symmetry from natural images to noise | `python scripts/jacobian_symmetry_sweep.py --device cuda:0` then `python scripts/plot_jacobian_sweep.py` | `results/jacobian_symmetry_sweep.json`, `figures/jacobian_sweep.pdf` |
| Jacobian symmetry on clean, noisy and denoised images | `python scripts/jacobian_symmetry_full.py --device cuda:0 --clean-only --out results/jacobian_symmetry_full_clean.json` and `python scripts/jacobian_symmetry_full.py --device cuda:0` | `results/jacobian_symmetry_full*.json` |
| numbers of Section 3.4 (pairs, spline, eigenvalues) | `python scripts/paper_numbers.py --device cuda:0` | `results/paper_numbers.json` |

The CT scripts run in the ODL / ASTRA environment. The search starts from the optima reported in the paper, so each setting
takes about an hour on one GPU; starting from the defaults (`--lamb-init 30 --sigma-init 10`) reaches the same optima in
about two hours. The TV, CRR, WCRR and MFoE rows of Table 2 and the WCRR curve of Fig. 1 come from the MFoE and WCRR
papers and are not reproduced here; `results/ct/changes_60.json` keeps the WCRR curve as data.

## Layout

- `ncr/`: the model (`linear.py`, `activations.py`, `models.py`), the solvers (`solvers.py`), the fixed-point denoiser with
  implicit differentiation (`denoiser.py`), the training loop (`trainer.py`), data and utilities.
- `scripts/`: data preparation, evaluation, CT reconstruction, figures and numbers.
- `tests/run_tests.py`: checks of the activation, the field bounds, the solvers, the implicit gradient, a training smoke run
  and the shipped checkpoint.

## Citation

```bibtex
@article{zavala2026nonconservative,
  title   = {From Critical Points to Fixed Points: Non-Conservative Ridge Regularizers for Image Reconstruction},
  author  = {Zavala-Mondrag{\'o}n, Luis A. and Ducotterd, Stanislas},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
