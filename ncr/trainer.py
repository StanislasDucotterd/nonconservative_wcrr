"""Config-driven training of fixed-point denoisers (the CRR / WCRR protocol:
random 64x64 crops of the 400 BSD images with D4 augmentation, batch 64, Adam with exponential decay from
lr_start to lr_end over `epochs` epochs, L1 loss, validation on full BSD68 images with fixed noise)."""
import math
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import build_train_dataset, build_val_dataset, fixed_noise
from .models import build_field
from .denoiser import build_denoiser
from .utils import CSVLogger, save_json, psnr, seed_everything
from . import diagnostics

try:  # optional
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


class Trainer:
    # logged per training step (CSV and TensorBoard): loss, lr, Lipschitz bound, solver iterations and convergence,
    # skipped steps, active penalties; per validation: PSNR per noise level, iterations, certified modulus, activation stats
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device(device)
        seed_everything(cfg.get("seed", 0))
        self.sigma = float(cfg["sigma"]) / 255.0
        # training noise level: fixed `sigma`, or per sample ~ U(sigma_range) (0-255 units, as CRR / WCRR); validation at val_sigmas
        self.sigma_range = cfg.get("sigma_range", None)
        self.val_sigmas = [float(s) for s in (cfg.get("val_sigmas", None) or [cfg["sigma"]])]
        dcfg = cfg["data"]
        tcfg = cfg["train"]
        self.train_set = build_train_dataset(dcfg)
        self.val_set = build_val_dataset(dcfg)
        self.train_loader = DataLoader(self.train_set, batch_size=dcfg.get("batch_size", 64), shuffle=True,
                                       num_workers=dcfg.get("num_workers", 4), drop_last=True, persistent_workers=dcfg.get("num_workers", 4) > 0)
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=0)
        self.field = build_field(cfg["model"]).to(self.device)
        self.model = build_denoiser(self.field, cfg.get("denoiser", {})).to(self.device)
        self.epochs = int(tcfg["epochs"])
        self.val_every = int(dcfg.get("val_every", max(1, self.epochs // 10)))
        self.loss_name = tcfg.get("loss", "l1")
        self.pen_w = dict(tcfg.get("penalties", {}))
        self.grad_clip = tcfg.get("grad_clip", None)
        self.diag_every = int(tcfg.get("diag_every", 0) or 0)
        self.min_converged = float(tcfg.get("min_converged_frac", 0.9))  # skip the step if fewer samples converged
        self.skipped = 0
        self.consecutive_skips = 0
        # optimiser: per-group learning-rate multipliers
        lr = float(tcfg["lr_start"])
        mult = dict(tcfg.get("lr_mult", {}))
        groups = [{"params": ps, "lr": lr * float(mult.get(name, 1.0)), "name": name}
                  for name, ps in self.field.param_groups().items() if len(ps) > 0]
        self.optimizer = torch.optim.Adam(groups, lr=lr)
        gamma = (float(tcfg["lr_end"]) / lr) ** (1.0 / max(1, self.epochs))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        # logging
        self.exp_dir = os.path.join(cfg.get("log_dir", "exps"), cfg["exp_name"])
        os.makedirs(self.exp_dir, exist_ok=True)
        save_json(cfg, os.path.join(self.exp_dir, "config.json"))
        self.train_log = CSVLogger(os.path.join(self.exp_dir, "train_log.csv"))
        self.val_log = CSVLogger(os.path.join(self.exp_dir, "val_log.csv"))
        self.tb = SummaryWriter(os.path.join(self.exp_dir, "tensorboard_logs")) if (SummaryWriter is not None and tcfg.get("tensorboard", True)) else None
        self.step = 0
        self.start_epoch = 0
        self.best_psnr = -1.0
        with open(os.path.join(self.exp_dir, "model.txt"), "w") as f:
            f.write(self.field.summary() + "\n" + str(self.field) + "\n")
        print(self.field.summary())

    # ------------------------------------------------------------------ loss
    def loss_fn(self, out, target):
        if self.loss_name == "l1":
            return (out - target).abs().mean()
        if self.loss_name == "mse":
            return ((out - target) ** 2).mean()
        raise ValueError(self.loss_name)

    def penalty(self):
        pens = self.field.penalties()
        total = torch.zeros((), device=self.device)
        logs = {}
        for name, w in self.pen_w.items():
            if float(w) != 0.0 and name in pens:  # only active (weighted) penalties are logged
                total = total + float(w) * pens[name]
                logs[f"pen_{name}"] = float(pens[name])
        return total, logs

    # ------------------------------------------------------------------ loops
    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.train_epoch(epoch)
            self.scheduler.step()
            if (epoch + 1) % self.val_every == 0 or epoch + 1 == self.epochs:
                self.validate(epoch + 1)
            self.save("last.pth", epoch + 1)
        if self.tb:
            self.tb.flush()
            self.tb.close()

    def train_epoch(self, epoch):
        self.model.train()
        bar = tqdm(self.train_loader, ncols=120, leave=False, desc=f"epoch {epoch}")
        for data in bar:
            data = data.to(self.device, non_blocking=True)
            B = data.shape[0]
            if self.sigma_range:
                sigma255 = torch.empty(B, device=self.device).uniform_(float(self.sigma_range[0]), float(self.sigma_range[1]))
            else:
                sigma255 = torch.full((B,), self.sigma * 255.0, device=self.device)
            noisy = data + (sigma255 / 255.0).view(B, 1, 1, 1) * torch.randn_like(data)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(noisy, sigma=sigma255)
            fid = self.loss_fn(out, data)
            pen, pen_logs = self.penalty()
            loss = fid + pen
            stats = self.model.stats()
            skipped = 0
            if stats.get("fp_converged_frac", 1.0) < self.min_converged:
                # the implicit gradient is only valid at a fixed point: do not update on a stalled solve
                skipped = 1
                self.skipped += 1
                self.consecutive_skips += 1
            else:
                self.consecutive_skips = 0
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.field.parameters(), float(self.grad_clip))
                self.optimizer.step()
            row = {"step": self.step, "epoch": epoch, "loss": float(fid), "lr": self.optimizer.param_groups[0]["lr"],
                   "lip_bound": float(self.field.lip_bound()), "fp_iters_mean": stats.get("fp_iters_mean", 0.0),
                   "fp_iters_max": stats.get("fp_iters_max", 0.0), "fp_converged_frac": stats.get("fp_converged_frac", 1.0),
                   "skipped": skipped}
            row.update(pen_logs)
            self.train_log.log(row)
            if self.tb:
                for k, v in row.items():
                    if k not in ("step", "epoch"):
                        self.tb.add_scalar(f"train/{k}", v, self.step)
            bar.set_postfix(loss=f"{float(fid):.4f}", pen=f"{float(pen):.2e}", it=f"{row.get('fp_iters_mean', 0):.0f}")
            self.step += 1

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        seed = int(self.cfg["data"].get("val_noise_seed", 1234))
        ps = {s: [] for s in self.val_sigmas}
        its = []
        for i, data in enumerate(self.val_loader):
            data = data.to(self.device)
            for k, s in enumerate(self.val_sigmas):  # the first sigma keeps the historical noise seeds (seed + i)
                noisy = data + fixed_noise(data.shape, s / 255.0, seed + i + 100000 * k, device=self.device)
                out = self.model(noisy, sigma=torch.full((1,), s, device=self.device)).clamp(0, 1)
                ps[s].append(float(psnr(out, data).mean()))
                if k == 0:
                    its.append(self.model.stats().get("fp_iters_mean", 0.0))
        mean_psnr = sum(ps[self.val_sigmas[0]]) / len(ps[self.val_sigmas[0]])  # first noise level: selects best.pth
        row = {"epoch": epoch, "step": self.step}
        for s in self.val_sigmas:
            row[f"psnr_sigma{s:g}"] = sum(ps[s]) / len(ps[s])
        row.update({"fp_iters_mean": sum(its) / len(its), "guarantee": self.field.guarantee()})
        wb = self.field.weak_monotone_bound()
        row["hypo_bound"] = float(wb) if wb is not None else float("nan")
        cert = self.field.act.certificate()
        for k, v in cert.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                row[f"cert_{k}"] = v
        if self.diag_every and (epoch // self.val_every) % self.diag_every == 0:
            x = self.val_set[0][None].to(self.device)
            x = x[:, :, :128, :128]
            with torch.enable_grad():
                rep = diagnostics.field_report(self.field, x, n_probes=2, iters=20)
            row.update({f"diag_{k}": v for k, v in rep.items()})
        self.val_log.log(row)
        if self.tb:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k not in ("epoch", "step"):
                    self.tb.add_scalar(f"val/{k}", v, epoch)
            if type(self.field.act).__name__ == "LinearSplineActivation":  # figures of all the learned splines
                try:
                    import importlib
                    import ncr.plots as P
                    P = importlib.reload(P)  # plotting tweaks apply to running jobs at their next validation
                    import matplotlib.pyplot as plt
                    qs = P.input_quantiles(self.field, self.val_set[0][None].to(self.device))  # where the data lives
                    figs = [("activations/splines", P.spline_figure(self.field, ncols=8, data_quantiles=qs))]
                    if not getattr(self.field.act, "shared", False):  # shared spline: the single curve is already drawn
                        figs.append(("activations/splines_overlay", P.spline_overlay_figure(self.field)))
                    if self.field.conditioned:
                        figs.append(("activations/sigma_scaling", P.sigma_scaling_figure(self.field)))
                    for tag, fig in figs:
                        self.tb.add_figure(tag, fig, epoch)
                        plt.close(fig)
                except Exception as e:  # never let a plotting problem stop a run
                    print(f"[warn] spline figures failed: {type(e).__name__}: {e}")
        text = "  ".join(f"sigma{s:g}: {row[f'psnr_sigma{s:g}']:.3f} dB" for s in self.val_sigmas)
        print(f"[val] epoch {epoch} step {self.step} PSNR {text}  iters {row['fp_iters_mean']:.1f}  {self.field.guarantee()}")
        if mean_psnr > self.best_psnr:
            self.best_psnr = mean_psnr
            self.save("best.pth", epoch)
        self.model.train()

    # ------------------------------------------------------------------ checkpoints
    def save(self, name, epoch):
        torch.save({"epoch": epoch, "step": self.step, "best_psnr": self.best_psnr, "config": self.cfg,
                    "field": self.field.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()}, os.path.join(self.exp_dir, name))

    def resume(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.field.load_state_dict(ck["field"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scheduler.load_state_dict(ck["scheduler"])
        self.step = ck["step"]
        self.start_epoch = ck["epoch"]
        self.best_psnr = ck.get("best_psnr", -1.0)
        print(f"resumed from {path} at epoch {self.start_epoch}")


def load_checkpoint(path, device="cpu"):
    """Rebuild (field, denoiser, config) from a checkpoint."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    field = build_field(cfg["model"]).to(device)
    field.load_state_dict(ck["field"])
    model = build_denoiser(field, cfg.get("denoiser", {})).to(device).eval()
    return field, model, cfg
