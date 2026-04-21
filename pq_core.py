"""
pq_core.py – Core mathematical functions for (p,q)-calculus reduction project.
This module contains the exact stable implementations from the author's original codes.
All overflow protection mechanisms are preserved.
"""

import math
from math import log, exp, log1p, fsum
import numpy as np

# ============================================================================
# q-gamma function (logarithmic, stable)
# ============================================================================

def q_gamma(z, q, max_iter=10000, tol=1e-14):
    """
    Γ_q(z) with logarithmic stability (exactly as in original codes)
    """
    if z <= 0:
        raise ValueError("z must be positive")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0,1)")
    if 1.0 - q < 1e-12:
        from math import gamma
        return gamma(z)
    if abs(z - 1.0) < 1e-14:
        return 1.0

    log_result = (1.0 - z) * log1p(-q)
    acc_logs = []
    for k in range(max_iter):
        r_num = q ** (k + 1)
        r_den = q ** (z + k)
        if r_num < tol and r_den < tol:
            break
        acc_logs.append(log1p(-r_num) - log1p(-r_den))
    log_sum = fsum(acc_logs)
    return exp(log_result + log_sum)


# ============================================================================
# (p,q)-gamma function (direct, logarithmic)
# ============================================================================

def pq_gamma(z, p, q, max_iter=20000, tol=1e-14):
    """
    Γ_{p,q}(z) with rigorous error control (exactly as in original codes)
    """
    if z <= 0:
        raise ValueError("z must be positive")
    if not (0.0 < q < p <= 1.0):
        raise ValueError("0 < q < p ≤ 1 required")
    if abs(z - 1.0) < 1e-14:
        return 1.0
    if abs(p - 1.0) < 1e-14:
        return q_gamma(z, q, max_iter=max_iter, tol=tol)

    r = q / p
    log_const = (z - 1.0) * (log(p) - log(p - q))
    logs = []
    for k in range(max_iter):
        r_num = r ** (k + 1)
        r_den = r ** (z + k)
        if r_num < tol and r_den < tol:
            break
        logs.append(log1p(-r_num) - log1p(-r_den))
    log_sum = fsum(logs)
    return exp(log_const + log_sum)


# ============================================================================
# (p,q)-Mittag-Leffler function (with full overflow protection)
# ============================================================================

def pq_mittag_leffler(alpha, beta, p, q, z, max_terms=500, tol=1e-14):
    """
    Stable computation of E_{α,β}^{p,q}(z) with overflow protection.
    Exactly matches the implementation in codeforsolvepqrelaxeqnumer.py
    """
    if abs(z) < 1e-10:
        return 1.0 / pq_gamma(beta, p, q)

    result = 1.0 / pq_gamma(beta, p, q)
    current_term = result

    for k in range(1, max_terms):
        prev_gamma_arg = alpha * (k-1) + beta
        curr_gamma_arg = alpha * k + beta

        try:
            if prev_gamma_arg > 50:
                log_ratio = alpha * log(k-1 + beta/alpha)
                gamma_prev = pq_gamma(prev_gamma_arg, p, q)
                gamma_curr = pq_gamma(curr_gamma_arg, p, q)
                if gamma_curr == 0 or not np.isfinite(gamma_prev) or not np.isfinite(gamma_curr):
                    ratio = exp(log_ratio)
                else:
                    ratio = gamma_prev / gamma_curr
            else:
                gamma_prev = pq_gamma(prev_gamma_arg, p, q)
                gamma_curr = pq_gamma(curr_gamma_arg, p, q)
                if gamma_curr == 0:
                    ratio = 0.0
                else:
                    ratio = gamma_prev / gamma_curr

            # Overflow protection with log space
            if abs(current_term) > 1e100 or abs(z * ratio) > 1e100:
                log_term = log(abs(current_term)) + log(abs(z)) + log(abs(ratio))
                if log_term > 700:
                    break
                else:
                    current_term = np.sign(current_term * z * ratio) * exp(log_term)
            else:
                current_term = current_term * z * ratio

            if np.isfinite(current_term):
                result += current_term
            else:
                break

            if abs(current_term) < tol * abs(result):
                break

        except (OverflowError, ValueError):
            break

    return result


def q_mittag_leffler(alpha, beta, r, z, max_terms=500, tol=1e-14):
    """
    Stable computation of E_{α,β}^{r}(z) with overflow protection.
    Exactly matches the implementation in codeforsolvepqrelaxeqnumer.py
    """
    if abs(z) < 1e-10:
        return 1.0 / q_gamma(beta, r)

    result = 1.0 / q_gamma(beta, r)
    current_term = result

    for k in range(1, max_terms):
        prev_gamma_arg = alpha * (k-1) + beta
        curr_gamma_arg = alpha * k + beta

        try:
            if prev_gamma_arg > 50:
                log_ratio = alpha * log(k-1 + beta/alpha)
                gamma_prev = q_gamma(prev_gamma_arg, r)
                gamma_curr = q_gamma(curr_gamma_arg, r)
                if gamma_curr == 0 or not np.isfinite(gamma_prev) or not np.isfinite(gamma_curr):
                    ratio = exp(log_ratio)
                else:
                    ratio = gamma_prev / gamma_curr
            else:
                gamma_prev = q_gamma(prev_gamma_arg, r)
                gamma_curr = q_gamma(curr_gamma_arg, r)
                if gamma_curr == 0:
                    ratio = 0.0
                else:
                    ratio = gamma_prev / gamma_curr

            # Overflow protection with log space
            if abs(current_term) > 1e100 or abs(z * ratio) > 1e100:
                log_term = log(abs(current_term)) + log(abs(z)) + log(abs(ratio))
                if log_term > 700:
                    break
                else:
                    current_term = np.sign(current_term * z * ratio) * exp(log_term)
            else:
                current_term = current_term * z * ratio

            if np.isfinite(current_term):
                result += current_term
            else:
                break

            if abs(current_term) < tol * abs(result):
                break

        except (OverflowError, ValueError):
            break

    return result