"""Cox proportional-hazards model (numpy/scipy only).

    log-partial-likelihood (Breslow ties), optional strata, L2 penalty
    Newton-Raphson with step halving
    model-based and cluster-robust (sandwich) standard errors
    Breslow baseline cumulative hazard -> absolute risk by day h

Breslow's tie handling is used throughout so the point estimates and the
robust variance are exactly consistent. With integer-day times and a small
event fraction per day (<0.3% of each risk set here), Breslow and Efron agree
closely; tests/test_pipeline.py cross-checks against lifelines when installed.
"""
import numpy as np
from scipy import stats


class CoxPH:
    def __init__(self, penalizer: float = 0.0, max_iter: int = 60, tol: float = 1e-9):
        self.penalizer = penalizer
        self.max_iter = max_iter
        self.tol = tol

    # ----------------------------------------------------------------- core --
    @staticmethod
    def _stratum_terms(X, time, event, eta, need_hess=True):
        """Log-likelihood, gradient, information for one stratum (Breslow)."""
        w = np.exp(eta - eta.max())                    # stable: the shift cancels in the ratios
        shift = eta.max()
        ut, inv = np.unique(time, return_inverse=True)
        k = len(ut)
        p = X.shape[1]
        # Per-time group sums, then reverse cumulative sums = risk-set sums.
        s0 = np.bincount(inv, weights=w, minlength=k)
        s1 = np.zeros((k, p))
        np.add.at(s1, inv, X * w[:, None])
        d = np.bincount(inv, weights=event, minlength=k)
        xe = np.zeros((k, p))
        np.add.at(xe, inv, X * event[:, None])
        S0 = np.cumsum(s0[::-1])[::-1]
        S1 = np.cumsum(s1[::-1], axis=0)[::-1]
        has = d > 0
        ll = float((eta * event).sum() - (d[has] * (np.log(S0[has]) + shift)).sum())
        xbar = S1 / S0[:, None]
        grad = (xe[has] - d[has, None] * xbar[has]).sum(axis=0)
        info = None
        if need_hess:
            info = np.zeros((p, p))
            # S2 per time from group sums of w x x^T (only times with events matter,
            # but the reverse cumsum needs every group).
            s2 = np.zeros((k, p, p))
            for j in range(k):
                m = inv == j
                Xm = X[m]
                s2[j] = (Xm * w[m, None]).T @ Xm
            S2 = np.cumsum(s2[::-1], axis=0)[::-1]
            for j in np.flatnonzero(has):
                info += d[j] * (S2[j] / S0[j] - np.outer(xbar[j], xbar[j]))
        return ll, grad, info, (ut, inv, S0, xbar, d, shift)

    def _objective(self, beta, need_hess=True):
        ll, g, H = 0.0, np.zeros_like(beta), np.zeros((len(beta), len(beta)))
        for idx in self._strata_idx:
            X, t, e = self._X[idx], self._t[idx], self._e[idx]
            l_, g_, H_, _ = self._stratum_terms(X, t, e, X @ beta, need_hess)
            ll += l_
            g += g_
            if need_hess:
                H += H_
        lam = self.penalizer
        ll -= 0.5 * lam * beta @ beta
        g -= lam * beta
        if need_hess:
            H += lam * np.eye(len(beta))
        return ll, g, H

    def fit(self, X, time, event, strata=None, cluster=None):
        X = np.asarray(X, float)
        self._X, self._t, self._e = X, np.asarray(time, float), np.asarray(event, float)
        strata = np.zeros(len(X), int) if strata is None else np.asarray(strata)
        self.strata_values_ = np.unique(strata)
        self._strata_idx = [np.flatnonzero(strata == s) for s in self.strata_values_]

        beta = np.zeros(X.shape[1])
        ll, g, H = self._objective(beta)
        self.converged_ = False
        for it in range(self.max_iter):
            step = np.linalg.solve(H, g)
            t = 1.0
            while True:
                nb = beta + t * step
                nll, ng, nH = self._objective(nb)
                if nll >= ll - 1e-12 or t < 1e-6:
                    break
                t /= 2
            done = abs(nll - ll) < self.tol * (1 + abs(ll)) and np.max(np.abs(t * step)) < 1e-6
            beta, ll, g, H = nb, nll, ng, nH
            if done:
                self.converged_ = True
                break
        self.n_iter_ = it + 1
        self.coef_ = beta
        self.loglik_ = ll
        self.information_ = H
        cov = np.linalg.inv(H)
        self.se_ = np.sqrt(np.diag(cov))
        self.cov_ = cov
        if cluster is not None:
            U = self.score_residuals()
            Uc = _sum_by(U, np.asarray(cluster))
            self.cov_robust_ = cov @ (Uc.T @ Uc) @ cov
            self.robust_se_ = np.sqrt(np.diag(self.cov_robust_))
        self._baseline()
        del self._X, self._t, self._e
        return self

    # ------------------------------------------------------------ residuals --
    def score_residuals(self):
        """Per-observation score residuals (Breslow), used for the robust variance."""
        X, t, e, beta = self._X, self._t, self._e, self.coef_
        U = np.zeros_like(X)
        for idx in self._strata_idx:
            Xs, ts, es = X[idx], t[idx], e[idx]
            eta = Xs @ beta
            _, _, _, (ut, inv, S0, xbar, d, shift) = self._stratum_terms(Xs, ts, es, eta, need_hess=False)
            w = np.exp(eta - shift)
            a = np.cumsum(np.where(d > 0, d / S0, 0.0))                      # sum_{t_k<=t} d_k/S0_k
            b = np.cumsum(np.where(d[:, None] > 0, d[:, None] * xbar / S0[:, None], 0.0), axis=0)
            U[idx] = es[:, None] * (Xs - xbar[inv]) - w[:, None] * (Xs * a[inv][:, None] - b[inv])
        return U

    # ------------------------------------------------------------- baseline --
    def _baseline(self):
        self.baseline_ = {}
        for s, idx in zip(self.strata_values_, self._strata_idx):
            Xs, ts, es = self._X[idx], self._t[idx], self._e[idx]
            eta = Xs @ self.coef_
            _, _, _, (ut, inv, S0, xbar, d, shift) = self._stratum_terms(Xs, ts, es, eta, need_hess=False)
            H0 = np.cumsum(d / (S0 * np.exp(shift)))
            self.baseline_[s] = (ut, H0)

    def cumulative_baseline_hazard(self, h, stratum=None):
        s = self.strata_values_[0] if stratum is None else stratum
        ut, H0 = self.baseline_[s]
        i = np.searchsorted(ut, h, side="right") - 1
        return 0.0 if i < 0 else float(H0[i])

    # ------------------------------------------------------------ prediction --
    def predict_log_hazard(self, X):
        return np.asarray(X, float) @ self.coef_

    def predict_risk(self, X, h, stratum=None):
        """P(event by day h) = 1 - exp(-H0(h) * exp(eta))."""
        return 1 - np.exp(-self.cumulative_baseline_hazard(h, stratum) * np.exp(self.predict_log_hazard(X)))

    def summary(self, names, robust=True):
        se = self.robust_se_ if robust and hasattr(self, "robust_se_") else self.se_
        z = self.coef_ / se
        return {
            "feature": list(names), "coef": self.coef_, "se": se,
            "hr": np.exp(self.coef_), "hr_lo": np.exp(self.coef_ - 1.96 * se),
            "hr_hi": np.exp(self.coef_ + 1.96 * se), "p": 2 * stats.norm.sf(np.abs(z)),
        }


def _sum_by(U, groups):
    _, g = np.unique(groups, return_inverse=True)
    out = np.zeros((g.max() + 1, U.shape[1]))
    np.add.at(out, g, U)
    return out
