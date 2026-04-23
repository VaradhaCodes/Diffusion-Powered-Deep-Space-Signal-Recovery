"""GMSK BER theory in AWGN.

REF: Murota & Hirade 1981, "GMSK Modulation for Digital Mobile Radio Telephony"
     IEEE Trans. Comm. COM-29(7) pp.1044-1050. DOI: 10.1109/TCOM.1981.1095108
     Table II: BER vs Eb/N0 for BT = 0.3 and 0.5.

Approximation used (Murota & Hirade 1981, Eq. (11)):
  BER ≈ Q(sqrt(2 * η(BT) * Eb/N0))

where η(BT) is the modulation efficiency factor (< 1 due to ISI from Gaussian filter),
calibrated from Table II in Murota & Hirade 1981.
"""

import math
from scipy.special import erfc

from src.physics.constants import GMSK_ETA


def _qfunc(x: float) -> float:
    """Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(x / math.sqrt(2))


def gmsk_awgn_ber(ebn0_db: float, bt: float) -> float:
    """GMSK BER in AWGN for a given Eb/N0 and BT product.

    REF: Murota & Hirade 1981, Eq. (11) and Table II.
      BER ≈ Q(sqrt(2 * η * Eb/N0))

    Args:
        ebn0_db : Eb/N0 in dB
        bt      : BT product (0.3 or 0.5; others use nearest value)
    """
    # find nearest supported BT
    supported = list(GMSK_ETA.keys())
    bt_key = min(supported, key=lambda x: abs(x - bt))
    eta = GMSK_ETA[bt_key]
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)
    return _qfunc(math.sqrt(2.0 * eta * ebn0_linear))
