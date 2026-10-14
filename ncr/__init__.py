"""ncr: non-conservative ridge regularizers, fixed-point denoisers and reconstruction.

Modules
-------
data         HDF5 image datasets (one image per key) with random crops and D4 augmentation
linear       Multi-layer convolution W with spectral normalisation (and the perturbed last layer used for V)
activations  The learnable linear spline of CRR / WCRR with the WCRR noise-level scaling
models       RidgeField M(x) = V^T phi(W x) with its Lipschitz bound
solvers      Batched fixed-point solvers: SAGD (accelerated gradient with adaptive restart) and Anderson acceleration
denoiser     Fixed-point denoiser x = y - M(x) trained by implicit differentiation
diagnostics  Jacobian asymmetry, monotonicity margin, Lipschitz estimates (matrix-free)
plots        TensorBoard figures of the learned splines
trainer      Config-driven training loop with CSV / TensorBoard logging
"""
__version__ = "1.0.0"
