"""Batched fixed-point and linear solvers.

Conventions
-----------
All operators take (x, idx) where idx are the indices (into the batch) of the samples in x, so that
per-sample early stopping is possible. For the denoising equation x + F(x) = y we use
    G(x, idx) = x + F(x) - y[idx]           (residual operator, strongly monotone if F is monotone)
    T(x, idx) = x - alpha * G(x, idx)       (damped fixed-point map)

Stopping rule ('stop' argument of solve_fixed_point):
    'change'  : ||x_{k+1} - x_k|| / ||x_{k+1}|| <= tol, the criterion of the CRR / WCRR code (used in the paper);
    'residual': per-sample fixed-point residual ||G(x)|| / scale <= tol, with scale = ||y||.
"""
import math
import torch


def bnorm(x):
    return x.flatten(1).norm(dim=1)


def _rel(new, old):
    return bnorm(new - old) / (bnorm(new) + 1e-8)


class SolveInfo(dict):
    """dict with attribute access: iterations (B,), residual (B,), converged (B,), history (list)."""

    def __getattr__(self, k):
        return self[k]


# ---------------------------------------------------------------- fixed-point solvers
def _default_scale(x0):
    sc = bnorm(x0)
    return torch.where(sc > 0, sc, torch.ones_like(sc))


def fista(G, x0, alpha, tol=1e-4, max_iter=200, restart=True, scale=None, stop="residual"):
    """Accelerated gradient descent on the forward map (method name 'agd'; 'fista' is the legacy name, there is no
    proximal step): x_{k+1} = z_k - alpha G(z_k), Nesterov momentum t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2,
    z_{k+1} = x_{k+1} + (t_k - 1) / t_{k+1} (x_{k+1} - x_k), and the gradient-based adaptive restart of O'Donoghue & Candes
    (t = 1, z = x whenever <G(z_k), x_{k+1} - x_k> > 0), per sample: line for line the solver of the CRR / WCRR code.
    Guaranteed for G = gradient of a convex function; heuristic otherwise (works on our nearly-symmetric untied fields).
    Residual stopping uses ||G(z_k)|| (the extrapolated point), which is available for free."""
    x = x0.clone()
    z = x0.clone()
    B = x.shape[0]
    scale = _default_scale(x0) if scale is None else scale
    t = torch.ones(B, device=x.device).view(-1, 1, 1, 1)
    idx = torch.arange(B, device=x.device)
    res = torch.ones(B, device=x.device, dtype=x.dtype)
    its = torch.zeros(B, device=x.device)
    hist = []
    for k in range(max_iter):
        x_old = x.clone()
        g = alpha * G(z[idx], idx)
        x[idx] = z[idx] - g
        t_old = t.clone()
        t = 0.5 * (1 + torch.sqrt(1 + 4 * t ** 2))
        z[idx] = x[idx] + (t_old[idx] - 1) / t[idx] * (x[idx] - x_old[idx])
        if stop == "residual":
            res[idx] = bnorm(g) / (alpha * scale[idx])
        elif k > 0:
            res[idx] = _rel(x[idx], x_old[idx])
        its[idx] += 1
        if restart:
            esti = torch.sum(g * (x[idx] - x_old[idx]), dim=(1, 2, 3))
            r = idx[(esti > 0).nonzero().view(-1)]
            t[r] = 1
            z[r] = x[r]
        idx = (res > tol).nonzero().view(-1)
        hist.append(res.max().item())
        if idx.numel() == 0:
            break
    return x, SolveInfo(iterations=its, residual=res, converged=res <= tol, history=hist)


agd = fista  # canonical name


def anderson(T, x0, m=5, lam=1e-6, beta=1.0, tol=1e-4, max_iter=200, alpha=1.0, scale=None, stop="residual", safeguard=True):
    """Type-II Anderson acceleration of x <- T(x) (batched; stops when the whole batch has converged).
    lam: Tikhonov regularisation relative to the mean squared residual norm; beta: mixing parameter.
    safeguard: if the accelerated step increases the residual of a sample, that sample takes the plain T step
    instead and its memory is restarted (cheap globalisation; without it Anderson can stagnate far from a fixed point
    on ill-conditioned non-symmetric problems).
    Robustness: the small (m+1)x(m+1) least-squares systems are solved in float64 and never raise (a singular system
    gets a pseudo-inverse solution); a sample whose iterate becomes non-finite is frozen at its last finite pair,
    reported with an infinite residual (converged=False) and excluded from the stopping rule, so one bad sample can
    neither crash a run nor poison the loss."""
    B = x0.shape[0]
    n = x0[0].numel()
    scale = _default_scale(x0) if scale is None else scale
    idx = torch.arange(B, device=x0.device)
    X = torch.zeros(B, m, n, dtype=x0.dtype, device=x0.device)
    Fm = torch.zeros(B, m, n, dtype=x0.dtype, device=x0.device)
    X[:, 0] = x0.flatten(1)
    Fm[:, 0] = T(x0, idx).flatten(1)
    X[:, 1] = Fm[:, 0]
    Fm[:, 1] = T(Fm[:, 0].view_as(x0), idx).flatten(1)
    H = torch.zeros(B, m + 1, m + 1, dtype=torch.float64, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    rhs = torch.zeros(B, m + 1, 1, dtype=torch.float64, device=x0.device)
    rhs[:, 0] = 1
    eye = torch.eye(m, dtype=torch.float64, device=x0.device)

    def crit(Fk, Xk, sc):
        d = (Fk - Xk).norm(dim=1)
        return d / (alpha * sc) if stop == "residual" else d / (1e-8 + Fk.norm(dim=1))

    def finite(rows):
        return torch.isfinite(rows).all(dim=1)

    res = crit(Fm[:, 1], X[:, 1], scale)
    dead = ~(finite(X[:, 1]) & finite(Fm[:, 1]) & torch.isfinite(res))
    last_x, last_f = X[:, 1].clone(), Fm[:, 1].clone()
    if dead.any():  # non-finite from the first applications of T: fall back to the input itself
        last_x[dead] = X[dead, 0]
        last_f[dead] = X[dead, 0]
        res[dead] = float("inf")
    prev = res.clone()
    hist = []
    k = 1
    for k in range(2, max_iter + 2):
        nk = min(k, m)
        Gm = Fm[:, :nk] - X[:, :nk]
        GG = torch.bmm(Gm, Gm.transpose(1, 2)).double()
        sc = GG.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-300).view(B, 1, 1)
        H[:, 1 : nk + 1, 1 : nk + 1] = GG + lam * sc * eye[:nk, :nk]
        sol, info = torch.linalg.solve_ex(H[:, : nk + 1, : nk + 1], rhs[:, : nk + 1])
        if bool((info != 0).any()):  # singular system for some samples: pseudo-inverse solution instead of raising
            bad_k = info != 0
            sol[bad_k] = torch.linalg.pinv(H[bad_k, : nk + 1, : nk + 1]) @ rhs[bad_k, : nk + 1]
        coef = sol[:, 1 : nk + 1, 0].to(x0.dtype)
        prev_slot = (k - 1) % m
        x_acc = beta * (coef[:, None] @ Fm[:, :nk])[:, 0] + (1 - beta) * (coef[:, None] @ X[:, :nk])[:, 0]
        nf = ~finite(x_acc)
        if nf.any():  # non-finite combination: plain step instead
            x_acc[nf] = Fm[nf, prev_slot]
        slot = k % m
        X[:, slot] = x_acc
        Fm[:, slot] = T(x_acc.view_as(x0), idx).flatten(1)
        res = crit(Fm[:, slot], X[:, slot], scale)
        nf = ~(torch.isfinite(res) & finite(Fm[:, slot]))
        bad = ((res > prev) | nf) if safeguard else nf
        if bad.any():
            # the accelerated step made things worse (or blew up): take the plain step from the previous iterate
            x_plain = Fm[bad, prev_slot]  # = T(previous iterate)
            X[bad, slot] = x_plain
            Fm[bad, slot] = T(x_plain.view_as(x0[bad]), idx[bad]).flatten(1)
            res[bad] = crit(Fm[bad, slot], X[bad, slot], scale[bad])
            # restart the memory of those samples: fill every slot with the current pair, so that the next
            # least-squares combination (identical rows -> uniform coefficients) is the plain step T(x)
            keep_x, keep_f = X[bad, slot].clone(), Fm[bad, slot].clone()
            X[bad] = keep_x[:, None, :]
            Fm[bad] = keep_f[:, None, :]
            nf = ~(torch.isfinite(res) & finite(Fm[:, slot]))
        if nf.any() or dead.any():
            dead = dead | nf
            X[dead] = last_x[dead][:, None, :]  # dead samples: frozen at their last finite pair, never updated again
            Fm[dead] = last_f[dead][:, None, :]
            res[dead] = float("inf")
        alive = ~dead
        last_x[alive] = X[alive, slot]
        last_f[alive] = Fm[alive, slot]
        prev = res.clone()
        hist.append(float(res[alive].max()) if bool(alive.any()) else float("inf"))
        if (not bool(alive.any())) or res[alive].max() < tol:
            break
    x = Fm[:, k % m].view_as(x0)  # return T(x_k) (one more step); the last finite T(x) for dead samples
    its = torch.full((B,), float(k), device=x0.device)
    return x, SolveInfo(iterations=its, residual=res, converged=res <= tol, history=hist)


def solve_fixed_point(method, G, T, x0, alpha, tol, max_iter, scale=None, stop="residual", **kw):
    """Dispatch. G is the residual operator, T = I - alpha G the damped map. scale: per-sample ||y|| for the
    residual criterion (defaults to ||x0||)."""
    common = dict(scale=scale, stop=stop)
    if method in ("agd", "fista"):  # accelerated gradient descent with Nesterov momentum and adaptive restart (SAGD)
        return fista(G, x0, alpha, tol=tol, max_iter=max_iter, restart=kw.get("restart", True), **common)
    if method == "anderson":
        return anderson(T, x0, m=kw.get("m", 5), lam=kw.get("lam", 1e-6), beta=kw.get("beta", 1.0), tol=tol,
                        max_iter=max_iter, alpha=alpha, safeguard=kw.get("safeguard", True), **common)
    raise ValueError(f"unknown fixed-point solver {method}: 'agd' (SAGD) or 'anderson'")


# ---------------------------------------------------------------- linear solver for implicit differentiation
def solve_adjoint(method, vjp, b, tol=1e-3, max_iter=50, **kw):
    """Solve (I - J^T) g = b given vjp(v) = J^T v: Anderson acceleration on g = J^T g + b (as in the WCRR code), or
    jfb (g = b, the Jacobian-free approximation)."""
    if method == "jfb":
        return b
    if method == "anderson":
        g, _ = anderson(lambda g, idx: vjp(g) + b, b, m=kw.get("m", 5), lam=kw.get("lam", 1e-6), beta=1.0, tol=tol, max_iter=max_iter, stop="change", safeguard=False)
        return g
    raise ValueError(f"unknown adjoint solver {method}: 'anderson' or 'jfb'")
