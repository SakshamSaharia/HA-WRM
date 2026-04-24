"""
HA-WRM vs WRM — Extended Analysis
==============================================
Onyeze, Banerjee, Fikioris, Tardos (SIGMETRICS 2025) + HA-WRM extension


CORRECTION VARIANTS IMPLEMENTED HERE
======================================
1. WRM             — Algorithm 2 from paper (exact, threshold requests)
2. HA-WRM-wr       — win-rate correction (given idea)
3. HA-WRM-wr-ema   — win-rate correction with EMA smoothing (α = 0.05)  (given idea)
    ema_wr_i[t] = (1-α)·ema_wr_i[t-1] + α·(wins_i[t]/t)
    correction_i = γ(t) · (p*/n - ema_wr_i[t])
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import beta as scipy_beta

BLUE   = "#2c7bb6"
RED    = "#d7191c"
GREEN  = "#1a9641"
C_BG1  = "#a6cee3"
C_BG2  = "#fdae61"
C_BG3  = "#b2df8a"

EMA_ALPHA = 0.05

# ════════════════════════════════════════════════════════════════
# 1.  DISTRIBUTIONS  (threshold-request aware, exact quantiles)
# ════════════════════════════════════════════════════════════════

class UniformDist:
    name = "Uniform[0,1]"
    def sample(self, rng, n):     return rng.uniform(0, 1, n)
    def quantile_threshold(self, p):
        """λ(p) = 1-p for Uniform: top-p fraction is (1-p, 1]."""
        return float(np.clip(1.0 - p, 0, 1))
    def V(self, p):
        p = float(np.clip(p, 1e-9, 1))
        return 1.0 - p / 2.0                     # E[V | V > 1-p]
    def U_sym(self, p, n):
        return self.V(p) * (1 - (1-p)**n) / n if p > 0 else 0.0
    def threshold_request(self, values, p_i, _rng):
        lam = np.array([self.quantile_threshold(float(p)) for p in p_i])
        return values > lam


class TwoPointDist:
    """P(V=high)=q, P(V=low)=1-q.  From Appendix C: low = q/(2+q)."""
    def __init__(self, q=0.25, high=1.0):
        self.q, self.high = q, high
        self.low  = q / (2 + q)
        self.name = f"Two-Point (q={q}, ε={self.low:.3f})"
    def sample(self, rng, n):
        return np.where(rng.uniform(size=n) < self.q, self.high, self.low)
    def quantile_threshold(self, p):
        return self.low if p <= 1 - self.q else self.high
    def V(self, p):
        p = float(np.clip(p, 1e-9, 1))
        if p <= self.q:
            return self.high
        return (self.q * self.high + (p - self.q) * self.low) / p
    def U_sym(self, p, n):
        return self.V(p) * (1 - (1-p)**n) / n if p > 0 else 0.0
    def threshold_request(self, values, p_i, rng):
        req = np.zeros(len(values), bool)
        for i, (v, p) in enumerate(zip(values, p_i)):
            p = float(np.clip(p, 0, 1))
            if p <= 0:   continue
            if p <= self.q:
                if v == self.high: req[i] = rng.uniform() < p / self.q
            else:
                if   v == self.high: req[i] = True
                elif v == self.low:  req[i] = rng.uniform() < (p-self.q)/(1-self.q)
        return req


class BimodalDist:
    """Equal mix of Beta(2,8) + Beta(8,2): two modes near 0.2 and 0.8."""
    name = "Bimodal (Beta(2,8)+Beta(8,2))"
    def __init__(self, n_grid=8000):
        g   = np.linspace(0, 1, n_grid)
        pdf = 0.5*scipy_beta(2,8).pdf(g) + 0.5*scipy_beta(8,2).pdf(g)
        dx  = g[1]-g[0]; pdf /= (pdf.sum()*dx)
        cdf = np.cumsum(pdf)*dx; cdf /= cdf[-1]
        self._g, self._pdf, self._cdf = g, pdf, cdf
    def sample(self, rng, n):
        return np.interp(rng.uniform(size=n), self._cdf, self._g)
    def quantile_threshold(self, p):
        """λ(p) = (1-p)-th quantile of the distribution."""
        return float(np.interp(np.clip(1-p, 0, 1), self._cdf, self._g))
    def V(self, p):
        p  = float(np.clip(p, 1e-9, 1))
        t  = self.quantile_threshold(p)
        mask = self._g >= t
        w  = self._pdf[mask]; s = w.sum()
        return float(np.dot(w, self._g[mask])/s) if s > 1e-12 else t
    def U_sym(self, p, n):
        return self.V(p)*(1-(1-p)**n)/n if p > 0 else 0.0
    def threshold_request(self, values, p_i, _rng):
        lam = np.array([self.quantile_threshold(float(p)) for p in p_i])
        return values > lam


class TruncExpDist:
    """Exp(λ) truncated to [0,1]."""
    def __init__(self, lam=3.0):
        self.lam = lam; self._Z = 1 - np.exp(-lam)
        self.name = f"Trunc. Exp. (λ={lam})"
    def sample(self, rng, n):
        return -np.log(1 - rng.uniform(size=n)*self._Z) / self.lam
    def quantile_threshold(self, p):
        """λ(p) = (1-p)-th quantile."""
        p = float(np.clip(1-p, 0, 1-1e-12))
        return float(-np.log(1 - p*self._Z) / self.lam)
    def V(self, p):
        p   = float(np.clip(p, 1e-9, 1))
        t   = self.quantile_threshold(p)
        lam = self.lam
        d   = np.exp(-lam*t) - np.exp(-lam)
        if d < 1e-14: return t
        return float(((t+1/lam)*np.exp(-lam*t) - (1+1/lam)*np.exp(-lam)) / d)
    def U_sym(self, p, n):
        return self.V(p)*(1-(1-p)**n)/n if p > 0 else 0.0
    def threshold_request(self, values, p_i, _rng):
        lam = np.array([self.quantile_threshold(float(p)) for p in p_i])
        return values > lam


# ════════════════════════════════════════════════════════════════
# 2.  HELPERS
# ════════════════════════════════════════════════════════════════

def Phi_inv(x, n):
    return 1.0 - (1.0 - float(np.clip(x, 0, 1-1e-15)))**(1.0/n)

def compute_p_star(dist, n, n_pts=5000):
    ps = np.linspace(1e-5, 1-1e-5, n_pts)
    Us = np.array([dist.U_sym(p, n) for p in ps])
    i  = int(np.argmax(Us))
    return float(ps[i]), float(Us[i])

def eta_paper(T, hi=0.8, lo=0.05, frac=0.10):
    """Paper Fig.2: linear decay from hi→lo over first frac·T rounds, constant thereafter."""
    t_end = max(1, int(frac*T))
    def f(t):
        return lo if t >= t_end else hi + (lo - hi)*t/t_end
    return f

def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode="valid") if w > 1 else x.copy()


# ════════════════════════════════════════════════════════════════
# 3.  SIMULATION ENGINE
# ════════════════════════════════════════════════════════════════

def simulate(algo, dist, n, T, p_star, eta_fn,
             gamma0=0.5, eps_res=0.02, beta_decay=0.55, seed=None):
    """
    Simulate DMMF for T rounds under one of three algorithms:
      'WRM'         – Algorithm 2 from paper (exact)
      'HA-WRM-wr'   – WRM + per-agent win-rate correction
      'HA-WRM-wr-ema' – WRM + per-agent win-rate correction with EMA smoothing

    Returns dict with:
      welfare_per_round  (T,)  – welfare earned each round
      welfare_cumul      (T,)  – cumulative welfare
      wins               (n,)  – total wins per agent
      M_hist             (T,)  – global WRM probability M[t]
      corr_hist          (n,T) – corrections (0 for WRM)
    """
    rng = np.random.default_rng(seed)

    K          = 0
    wins       = np.zeros(n, dtype=int)
    welfare    = np.zeros(n)
    welfare_pr = np.zeros(T)
    M_hist     = np.empty(T)
    corr_hist  = np.zeros((n, T))

    # Precompute targets for corrections
    V_star          = dist.V(p_star)
    target_win_rate = 0            # per agent per round (wins)
    ema_wr         = np.zeros(n, dtype=float)  # EMA of win-rates per agent

    for t in range(1, T+1):
        idx = t - 1
        target_win_rate = 0
        # ── Compute global WRM value M[t] ─────────────────────────
        if t == 1:
            M_t = p_star
            target_win_rate = (1-(1-p_star)**n)/ n               # per agent per round (wins)
        else:
            rate  = float(np.clip(K / (t-1), 0, 1-1e-12))
            # ζ(t) = 1  (paper Figure 2 empirical setup)
            target_win_rate = (1-(1-p_star)**n)/ n   
            M_t   = (1.0 - eta_fn(t)) * Phi_inv(rate, n) + eta_fn(t) * p_star
            M_t   = float(np.clip(M_t, 0, 1))
            
        M_hist[idx] = M_t

        # ── Per-agent probabilities ───────────────────────────────
        if algo == "WRM":
            p_i = np.full(n, M_t)

        else:
            # Correction decay schedule: γ(t) → ε_res > 0
            gamma_t = eps_res + (gamma0 - eps_res) / (1.0 + t**beta_decay)

            if t <= 1:
                corr = np.zeros(n)
            else:
                if algo == "HA-WRM-wr":
                    # Win-rate correction: δ_i = target_win_rate - wins_i / (t-1)
                    # Units: [probability]  ← directly comparable to p_i
                    actual_wr = wins / (t - 1)
                    corr = gamma_t * (target_win_rate - actual_wr)
                elif algo == "HA-WRM-wr-ema":
                    # Win-rate correction with EMA smoothing, alpha = 0.05
                    # ema_wr[t] = (1-alpha) * ema_wr[t-1] + alpha * (wins_i[t-1] / (t-1))
                    actual_wr = wins / (t - 1)
                    ema_wr = (1.0 - EMA_ALPHA) * ema_wr + EMA_ALPHA * actual_wr
                    corr = gamma_t * (target_win_rate - ema_wr)
                else:
                    corr = np.zeros(n)

            corr_hist[:, idx] = corr
            p_i = np.clip(M_t + corr, 0, 1)

        # ── Sample values & threshold-request ────────────────────
        values   = dist.sample(rng, n)
        requests = dist.threshold_request(values, p_i, rng)

        # ── DMMF allocation: fewest-wins requestor wins ───────────
        req_idx = np.where(requests)[0]
        if len(req_idx):
            w                   = req_idx[np.argmin(wins[req_idx])]
            wins[w]            += 1
            welfare[w]         += values[w]
            welfare_pr[idx]     = values[w]
            K                  += 1

    cumul = np.cumsum(welfare_pr)
    return dict(welfare_per_round=welfare_pr,
                welfare_cumul=cumul,
                wins=wins, welfare_final=welfare,
                M_hist=M_hist, corr_hist=corr_hist)


def run_seeds(algo, dist, n, T, p_star, eta_fn,
              gamma0=0.5, eps_res=0.02, beta_decay=0.55,
              n_seeds=6, base_seed=42):
    return [simulate(algo, dist, n, T, p_star, eta_fn,
                     gamma0=gamma0, eps_res=eps_res,
                     beta_decay=beta_decay, seed=base_seed + s*17)
            for s in range(n_seeds)]


def welfare_rate_matrix(results, T):
    """Returns (n_seeds, T) matrix of cumulative welfare / t."""
    ts = np.arange(1, T+1)
    return np.stack([r["welfare_cumul"] / ts for r in results])


# ════════════════════════════════════════════════════════════════
# 4.  PLOTTING
# ════════════════════════════════════════════════════════════════

ALGO_STYLES = {
    "WRM":            (BLUE,  C_BG1, "--",  "WRM (paper)"),
    "HA-WRM-wr":      (RED,   C_BG2, "-",   "HA-WRM win-rate"),
    "HA-WRM-wr-ema":  (GREEN, C_BG3, "-.",  "HA-WRM win-rate EMA"),
}
PLOT_ALGOS = ["WRM", "HA-WRM-wr", "HA-WRM-wr-ema"]


def _plot_one_welfare(ax, results_dict, dist_name, p_star, n, T, opt, win):
    ts = np.arange(1, T+1)
    for algo in PLOT_ALGOS:
        if algo not in results_dict: continue
        col, bg, ls, lbl = ALGO_STYLES[algo]
        wr = welfare_rate_matrix(results_dict[algo], T)
        mm, ms = wr.mean(0), wr.std(0)
        t_s = ts[:len(smooth(mm, win))]
        ax.fill_between(t_s, smooth(mm-ms,win), smooth(mm+ms,win), color=bg, alpha=0.28)
        ax.plot(t_s, smooth(mm,win), color=col, lw=2.0, ls=ls, label=lbl)
    ax.axhline(opt, color="k", ls=":", lw=1.5, label=f"Opt = {opt:.4f}")
    ax.set(title=dist_name, xlabel="Round $t$", ylabel="Total welfare / $t$")
    ax.legend(fontsize=8); ax.grid(alpha=0.22)


def plot_main_welfare(all_res, dists, n, T, path):
    nd  = len(dists); win = max(1, T//300)
    fig, axes = plt.subplots(nd, 2, figsize=(16, 4.5*nd))
    if nd == 1: axes = axes[None, :]
    ts = np.arange(1, T+1)

    for row, d in enumerate(dists):
        p_s  = all_res[d.name]["p_star"]
        opt  = d.U_sym(p_s, n) * n
        rdat = all_res[d.name]

        _plot_one_welfare(axes[row, 0], rdat, d.name, p_s, n, T, opt, max(1, T//300))

        # Relative gain vs WRM for both HA variants
        ax2  = axes[row, 1]
        wrm_m = welfare_rate_matrix(rdat["WRM"], T).mean(0)
        for algo in ["HA-WRM-wr", "HA-WRM-wr-ema"]:
            if algo not in rdat:
                continue
            col, bg, ls, lbl = ALGO_STYLES[algo]
            ha_m  = welfare_rate_matrix(rdat[algo], T).mean(0)
            rel   = np.where(wrm_m > 1e-9, 100*(ha_m - wrm_m)/wrm_m, 0)
            sr    = smooth(rel, max(1, T//300))
            t_s   = ts[:len(sr)]
            ax2.plot(t_s, sr, color=col, lw=1.8, ls=ls, label=lbl)
        ax2.axhline(0, color="k", ls="--", lw=1.0)
        ax2.set(title=f"{d.name} – Relative gain over WRM (%)",
                xlabel="Round $t$", ylabel="(HA−WRM)/WRM (%)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.22)

    fig.suptitle(
        r"WRM vs HA-WRM-wr vs HA-WRM-wr-EMA — Total Welfare" "\n"
        rf"$n={n}$,  $T={T:,}$,  $\gamma_0={0.5}$,  "
        r"$\varepsilon_{res}=0.02$  (correction → const, not 0)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_M_convergence(all_res, dists, n, T, path):
    """M[t] trajectory: both algos converge to p*."""
    nd  = len(dists); win = max(1, T//300)
    ts  = np.arange(1, T+1)
    fig, axes = plt.subplots(nd, 3, figsize=(18, 3.8*nd))
    if nd == 1: axes = axes[None, :]

    for row, d in enumerate(dists):
        p_s  = all_res[d.name]["p_star"]
        rdat = all_res[d.name]
        for col, (algo, (col_c, bg, ls, lbl)) in enumerate(ALGO_STYLES.items()):
            if algo not in rdat: continue
            ax  = axes[row, col]
            M   = np.stack([r["M_hist"] for r in rdat[algo]])
            mm, ms = M.mean(0), M.std(0)
            t_s = ts[:len(smooth(mm, win))]
            ax.fill_between(t_s, smooth(mm-ms,win), smooth(mm+ms,win), color=bg, alpha=0.35)
            ax.plot(t_s, smooth(mm,win), color=col_c, lw=2.0, ls=ls, label=lbl)
            ax.axhline(p_s, color="red", ls="--", lw=2.0,
                       label=f"$p^*={p_s:.3f}$")
            if row == 0: ax.set_title(lbl, fontsize=10, fontweight="bold", color=col_c)
            ax.text(0.02,0.95, d.name, transform=ax.transAxes, fontsize=8, va="top", alpha=0.7)
            ax.set(ylabel="$M[t]$", ylim=[max(0,p_s-0.25), min(1,p_s+0.25)])
            ax.legend(fontsize=8); ax.grid(alpha=0.22)
        for ax in axes[-1]: ax.set_xlabel("Round $t$")

    fig.suptitle(r"$M[t]$ Convergence — all algorithms → $p^*$",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_early_late(all_res, dists, n, T, path):
    """Bar chart: early (t ≤ T/5) vs late (t ≥ 4T/5) welfare."""
    e_end   = T // 5
    l_start = 4 * T // 5
    nd  = len(dists)
    fig, axes = plt.subplots(1, nd, figsize=(5*nd, 5.5))
    if nd == 1: axes = [axes]

    for ax, d in zip(axes, dists):
        p_s = all_res[d.name]["p_star"]
        opt = d.U_sym(p_s, n) * n
        xs  = np.arange(2); bw = 0.25
        for bi, algo in enumerate(PLOT_ALGOS):
            if algo not in all_res[d.name]: continue
            col, bg, ls, lbl = ALGO_STYLES[algo]
            wr = welfare_rate_matrix(all_res[d.name][algo], T)  # (seeds, T)
            early = wr[:, :e_end].mean(axis=1)
            late  = wr[:, l_start:].mean(axis=1)
            mv, sv = early.mean(), early.std()
            ml, sl = late.mean(),  late.std()
            ax.bar(xs + bi*bw, [mv, ml], bw, color=col, alpha=0.82, label=lbl,
                   yerr=[sv, sl], error_kw=dict(capsize=4, elinewidth=1.5))
        ax.axhline(opt, color="k", ls=":", lw=2.0, label=f"Opt = {opt:.3f}")
        ax.set_xticks(xs + bw)
        ax.set_xticklabels([f"Early\n(t ≤ {e_end:,})", f"Late\n(t ≥ {l_start:,})"])
        ax.set(title=d.name, ylabel="Avg welfare / round")
        ax.legend(fontsize=8); ax.grid(alpha=0.22, axis="y")

    fig.suptitle("Early vs. Late Welfare\n"
                 "WRM vs HA-WRM-wr",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_correction_decay(all_res, dists, T, path):
    """Show how corrections decay: win-rate vs welfare corrections."""
    nd  = len(dists); win = max(1, T//200)
    ts  = np.arange(1, T+1)
    fig, axes = plt.subplots(1, nd, figsize=(5*nd, 5))
    if nd == 1: axes = [axes]

    for ax, d in zip(axes, dists):
        p_s = all_res[d.name]["p_star"]
        for algo in ["HA-WRM-wr", "HA-WRM-wr-ema"]:
            col, bg, ls, lbl = ALGO_STYLES[algo]
            mags = np.stack([np.abs(r["corr_hist"]).mean(0)
                             for r in all_res[d.name][algo]])
            mm, ms = mags.mean(0), mags.std(0)
            t_s = ts[:len(smooth(mm, win))]
            ax.semilogy(t_s, smooth(mm,win)+1e-9, color=col, lw=2, ls=ls, label=lbl)
        ax.axhline(0.02 * 0.5, color="gray", ls=":", lw=1.2,
                   label=r"$\approx\varepsilon_{res}\cdot \gamma_0 / 2$")
        ax.set(title=d.name, xlabel="Round $t$",
               ylabel="|correction| (log-scale)")
        ax.legend(fontsize=8); ax.grid(alpha=0.22, which="both")

    fig.suptitle(r"HA-WRM-wr Correction Magnitude",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_bimodal_long_run(dist, n, T_long, p_star, eta_fn, path, n_seeds=5):
    """
    Verify that the Bimodal 'gap' is finite-T noise:
    WRM converges to p* but slowly because U(p) is flat near p*.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ts = np.arange(1, T_long+1)
    win = max(1, T_long//200)

    for algo, (col, bg, ls, lbl) in ALGO_STYLES.items():
        if algo not in ["WRM", "HA-WRM-wr"]:
            continue   # just compare WRM vs HA-WRM-wr
        res = run_seeds(algo, dist, n, T_long, p_star, eta_fn, n_seeds=n_seeds)
        M   = np.stack([r["M_hist"] for r in res])
        mm, ms = M.mean(0), M.std(0)
        t_s = ts[:len(smooth(mm, win))]
        ax1.fill_between(t_s, smooth(mm-ms,win), smooth(mm+ms,win), color=bg, alpha=0.3)
        ax1.plot(t_s, smooth(mm,win), color=col, lw=2.0, ls=ls, label=lbl)
        wr = welfare_rate_matrix(res, T_long)
        wm, ws = wr.mean(0), wr.std(0)
        ax2.fill_between(t_s, smooth(wm-ws,win), smooth(wm+ws,win), color=bg, alpha=0.3)
        ax2.plot(t_s, smooth(wm,win), color=col, lw=2.0, ls=ls, label=lbl)

    opt = dist.U_sym(p_star, n) * n
    ax1.axhline(p_star, color="red", ls="--", lw=2.0, label=f"$p^*={p_star:.3f}$")
    ax2.axhline(opt,    color="k",   ls=":",  lw=1.5, label=f"Opt = {opt:.4f}")
    ax1.set(title="Bimodal — M[t] long run", xlabel="Round $t$", ylabel="M[t]",
            ylim=[max(0,p_star-0.2), min(1,p_star+0.2)])
    ax2.set(title="Bimodal — Welfare long run", xlabel="Round $t$",
            ylabel="Total welfare / $t$")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
    fig.suptitle(
        f"Bimodal — Long Run (T={T_long:,})\n"
        "Verifying WRM gap is finite-T noise: both converge to p* and Opt",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


def plot_eps_sensitivity(dist, n, T, p_star, eta_fn, path, n_seeds=5):
    """Show how ε_res choice affects final welfare gap and convergence speed."""
    eps_vals = [0.0, 0.005, 0.01, 0.02, 0.05]
    cmap     = plt.cm.plasma(np.linspace(0.15, 0.85, len(eps_vals)))
    ts = np.arange(1, T+1); win = max(1, T//300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    wrm_res = run_seeds("WRM", dist, n, T, p_star, eta_fn, n_seeds=n_seeds)
    wrm_wm  = welfare_rate_matrix(wrm_res, T).mean(0)
    ax1.plot(ts[:len(smooth(wrm_wm,win))], smooth(wrm_wm,win),
             color=BLUE, lw=2.5, ls="--", label="WRM (ε=N/A)")
    ax2.axhline(0, color=BLUE, ls="--", lw=1.5, label="WRM baseline")

    for eps_v, col in zip(eps_vals, cmap):
        res = run_seeds("HA-WRM-wr-ema", dist, n, T, p_star, eta_fn,
                        eps_res=eps_v, n_seeds=n_seeds)
        wm  = welfare_rate_matrix(res, T).mean(0)
        sm  = smooth(wm, win)
        ax1.plot(ts[:len(sm)], sm, color=col, lw=1.8,
                 label=fr"HA ($\varepsilon_{{res}}={eps_v}$)")
        rel = np.where(wrm_wm > 1e-9, 100*(wm - wrm_wm)/wrm_wm, 0)
        sr  = smooth(rel, win)
        ax2.plot(ts[:len(sr)], sr, color=col, lw=1.8,
                 label=fr"$\varepsilon_{{res}}={eps_v}$")

    opt = dist.U_sym(p_star, n) * n
    ax1.axhline(opt, color="k", ls=":", lw=1.5, label=f"Opt={opt:.3f}")
    ax1.set(title=f"ε_res Sensitivity — {dist.name}",
            xlabel="Round $t$", ylabel="Total welfare / $t$")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.22)
    ax2.set(title="Relative gain over WRM (%)",
            xlabel="Round $t$", ylabel="%")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.22)
    fig.suptitle(r"Effect of $\varepsilon_{res}$ on HA-WRM" "\n"
                 r"ε_res=0 → correction vanishes; ε_res>0 → persistent micro-optimization",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved → {path}")


# ════════════════════════════════════════════════════════════════
# 5.  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    N       = 5
    T       = 1000_00
    T_LONG  = 200_000      # for Bimodal gap verification
    N_SEEDS = 8
    GAMMA0  = 0.5
    EPS_RES = 0.9
    BETA    = 0.55
    OUT     = "./hawrm-outputs"

    print("=" * 72)
    print("  HA-WRM vs WRM — Full Analysis")
    print("  Implementation of Onyeze et al. 2025 + HA-WRM extension")
    print("=" * 72)
    print(f"  n={N}  T={T:,}  seeds={N_SEEDS}  γ₀={GAMMA0}  ε_res={EPS_RES}  β={BETA}")

    dists  = [UniformDist(), TwoPointDist(q=0.25), BimodalDist(), TruncExpDist(lam=3.0)]
    eta_fn = eta_paper(T)

    # ── p* for each distribution ─────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  {'Distribution':<40}  {'p*':>6}  {'V(p*)':>7}  {'Opt welfare':>12}")
    print(f"  {'─'*68}")
    p_star_map = {}
    for d in dists:
        p_s, U_s = compute_p_star(d, N)
        p_star_map[d.name] = p_s
        print(f"  {d.name:<40}  {p_s:>6.4f}  {d.V(p_s):>7.4f}  {U_s*N:>12.4f}")

    # ── Run simulations ───────────────────────────────────────────
    print(f"\n  Running simulations ({N_SEEDS} seeds each)…")
    ALGOS = ["WRM", "HA-WRM-wr", "HA-WRM-wr-ema"]
    all_res = {}

    for d in dists:
        p_s = p_star_map[d.name]
        all_res[d.name] = {"p_star": p_s}
        for algo in ALGOS:
            print(f"    [{algo:14s}] {d.name} …", end="  ", flush=True)
            res = run_seeds(algo, d, N, T, p_s, eta_fn,
                            gamma0=GAMMA0, eps_res=EPS_RES,
                            beta_decay=BETA, n_seeds=N_SEEDS)
            all_res[d.name][algo] = res
            wf = np.mean([r["welfare_cumul"][-1]/T for r in res])
            print(f"welfare/round = {wf:.5f}")

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  WELFARE SUMMARY  (average over {N_SEEDS} seeds, final T rounds)")
    print(f"{'─' * 72}")
    print(f"  {'Distribution':<34}  {'WRM':>8}  {'HA-wr':>8}  {'Δwr%':>8}  {'Opt':>8}")
    print(f"  {'─'*68}")
    for d in dists:
        p_s = p_star_map[d.name]; opt = d.U_sym(p_s, N) * N
        def mf(a): return np.mean([r["welfare_cumul"][-1]/T for r in all_res[d.name][a]])
        ww  = mf("WRM"); wwr = mf("HA-WRM-wr"); wema = mf("HA-WRM-wr-ema")
        print(f"  {d.name:<34}  {ww:>8.5f}  {wwr:>8.5f}  {wema:>8.5f}  "
              f"{100*(wwr-ww)/ww:>+8.2f}%  {100*(wema-ww)/ww:>+8.2f}%  {opt:>8.5f}")

    # ── Early vs late breakdown ────────────────────────────────────
    print(f"\n  EARLY (t ≤ {T//5:,}) vs LATE (t ≥ {4*T//5:,}) welfare")
    print(f"  {'─'*68}")
    print(f"  {'Distribution':<34}  {'Phase':<8}  {'WRM':>8}  {'HA-wr':>8}  {'EMA':>8}")
    print(f"  {'─'*68}")
    e_end = T // 5; l_start = 4 * T // 5
    for d in dists:
        for phase, sl in [("Early", slice(None, e_end)), ("Late", slice(l_start, None))]:
            def mphase(a):
                return np.mean([r["welfare_cumul"][sl.stop-1 if sl.stop else -1] /
                                (sl.stop or T) - (r["welfare_cumul"][sl.start or 0] /
                                max(sl.start or 1, 1))
                                for r in all_res[d.name][a]])
            # Simpler: mean over seed of mean over rounds in phase
            def mp(a):
                return np.mean([r["welfare_per_round"][sl].mean()
                                for r in all_res[d.name][a]])
            ww  = mp("WRM"); wwr = mp("HA-WRM-wr"); wema = mp("HA-WRM-wr-ema")
            print(f"  {d.name:<34}  {phase:<8}  {ww:>8.5f}  {wwr:>8.5f}  {wema:>8.5f}")
    print(f"{'=' * 72}")

    # ── WRM M[t] convergence check ────────────────────────────────
    print(f"\n  WRM M[t] CONVERGENCE CHECK (final 10% of rounds)")
    print(f"  {'─'*60}")
    for d in dists:
        p_s = p_star_map[d.name]
        M_ends = [r["M_hist"][-T//10:].mean() for r in all_res[d.name]["WRM"]]
        m_mu, m_std = np.mean(M_ends), np.std(M_ends)
        print(f"  {d.name:<40}  M_final={m_mu:.4f}±{m_std:.4f}  |M-p*|={abs(m_mu-p_s):.4f}")

    # ── Generate all plots ────────────────────────────────────────
    print("\n  Generating plots…")
    plot_main_welfare   (all_res, dists, N, T,
                         f"{OUT}/hawrm_welfare_final.png")
    plot_M_convergence  (all_res, dists, N, T,
                         f"{OUT}/hawrm_M_convergence.png")
    plot_early_late     (all_res, dists, N, T,
                         f"{OUT}/hawrm_early_late.png")
    plot_correction_decay(all_res, dists, T,
                          f"{OUT}/hawrm_corrections.png")

    # Long-run verification of Bimodal gap
    print(f"  Running Bimodal long-run (T={T_LONG:,}) gap verification…")
    bim = BimodalDist()
    eta_long = eta_paper(T_LONG)
    # plot_bimodal_long_run(bim, N, T_LONG, p_star_map[bim.name],
    #                       eta_long, f"{OUT}/hawrm_bimodal_longrun.png", n_seeds=4)

    print(f"\n  ✓ All results saved to {OUT}/")
    print("\n" + "="*72)
    print("  DIAGNOSIS SUMMARY")
    print("="*72)
    
