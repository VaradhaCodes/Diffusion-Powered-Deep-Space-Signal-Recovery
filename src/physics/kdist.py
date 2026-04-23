"""K-distribution physics: moments and envelope PDF.

REF: Jao 1984, "Amplitude distribution of composite terrain scattered and
     line-of-sight signal and its application to land mobile radio channels"
     IEEE Trans. Vehicular Tech. VT-33(3) pp.59-68. DOI: 10.1109/TVT.1984.23517

K-distribution compound model (envelope amplitude r = |h|):
  G  ~ Gamma(alpha, b²/alpha)   → E[G] = b²
  r  ~ Rayleigh(sqrt(G/2))      → E[r²] = G  →  E[|h|²] = b²

The envelope PDF is (Jao 1984, Eq. (4); also Ward et al. IET 2006 Ch.4 Eq.4.19):
  p(r) = (4 / (Γ(α) * b^α)) * (r/b)^α * K_{α-1}(2r/b)     r ≥ 0

where K_{α-1} is the modified Bessel function of the second kind, order α-1.
Note: b here is the scale so that E[|h|²] = b² matches our generator convention.

Moments (Jao 1984, Eq. (5)):
  E[|h|²] = b²                        (second moment)
  E[|h|⁴] = b⁴ * (α + 1) / α        (fourth moment)
"""

import math
import numpy as np
from scipy.special import gamma as gamma_func, kv as bessel_kv

from src.physics.constants import K_DIST_SECOND_MOMENT_SCALE_EXPONENT


def kdist_second_moment(alpha: float, b: float) -> float:
    """E[|h|²] = b²  (independent of alpha).

    REF: Jao 1984, Eq. (5) first moment of |h|².
    """
    return float(b ** K_DIST_SECOND_MOMENT_SCALE_EXPONENT)


def kdist_fourth_moment(alpha: float, b: float) -> float:
    """E[|h|⁴] = b⁴ * (alpha + 1) / alpha.

    REF: Jao 1984, Eq. (5); Ward et al. IET 2006, Ch.4 Eq.(4.20).
    """
    return float(b**4 * (alpha + 1) / alpha)


def kdist_envelope_pdf(r: np.ndarray, alpha: float, b: float) -> np.ndarray:
    """K-distribution envelope PDF evaluated at radii r.

    REF: Jao 1984, Eq. (4):
      p(r) = (4 / (Γ(α) * b^α)) * (r/b)^α * K_{α-1}(2r/b)

    Args:
        r     : array of non-negative envelope values
        alpha : shape parameter (Zhu's α ∈ {5, 10})
        b     : scale (E[|h|²] = b²; for Zhu's channel b=2)
    """
    r = np.asarray(r, dtype=np.float64)
    order = alpha - 1.0
    # avoid division by zero at r=0
    r_safe = np.where(r > 1e-12, r, 1e-12)
    x = 2.0 * r_safe / b
    pdf = (4.0 / (gamma_func(alpha) * b**alpha)) * (r_safe / b)**alpha * bessel_kv(order, x)
    pdf = np.where(r > 1e-12, pdf, 0.0)
    return pdf
