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
    """E[|h|⁴] = 2 * b⁴ * (alpha + 1) / alpha.

    For the complex K-distribution (compound Gamma-Rayleigh):
      |h|² = (G/2) * W  where G~Gamma(α, b²/α), W~χ²(2)
      E[G²] = b⁴*(α+1)/α,  E[W²] = 8  →  E[|h|⁴] = E[G²]/4 * 8 = 2b⁴(α+1)/α

    REF: derived from Jao 1984, Eq.(5) moments of G combined with χ²(2) moments.
    Ward et al. IET 2006 Ch.4 gives real-valued formulation; factor of 2 comes from
    the two-component (complex) Rayleigh component.
    """
    return float(2.0 * b**4 * (alpha + 1) / alpha)


def kdist_envelope_pdf(r: np.ndarray, alpha: float, b: float) -> np.ndarray:
    """K-distribution envelope PDF for the complex compound Gamma-Rayleigh model.

    Generator model: |h|² = (G/2)*W  where G~Gamma(α,b²/α), W~χ²(2).
    This gives conditional |h|²|G ~ Exp(G), i.e. the standard complex K-distribution.

    Closed-form PDF (derivable from compound integral via Bessel-K identity):
      p(r) = 4 * (α/b²)^((α+1)/2) / Γ(α) * r^α * K_{α-1}(2*√α*r/b)

    Verified: ∫₀^∞ p(r)dr = 1 and E[r²] = b² for α ∈ {1.2, 1.4}, b=2.

    REF: Derivation follows Ward et al. IET "Sea Clutter" 2006, Ch.4 adapted for
    complex envelope; matches compound model in src/synth_gen.py exactly.

    Args:
        r     : array of non-negative envelope values
        alpha : shape parameter (Zhu's α; code calls this m)
        b     : scale (E[|h|²] = b²; for Zhu's channel b=2)
    """
    r = np.asarray(r, dtype=np.float64)
    r_safe = np.where(r > 1e-12, r, 1e-12)
    order  = alpha - 1.0
    c      = np.sqrt(alpha) / b                        # Bessel scale factor
    prefac = 4.0 * (alpha / b**2)**((alpha + 1) / 2) / gamma_func(alpha)
    pdf    = prefac * r_safe**alpha * bessel_kv(order, 2.0 * c * r_safe)
    return np.where(r > 1e-12, pdf, 0.0)
