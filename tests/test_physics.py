"""Oracle physics unit tests for V6 Batch 2.

All references are cited inline. Tests must pass before any validation logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physics.gmsk_theory import gmsk_awgn_ber
from src.physics.kdist import kdist_second_moment


def test_gmsk_awgn_ber_bt05_at_10db():
    # REF: Murota & Hirade 1981, "GMSK Modulation for Digital Mobile Radio Telephony"
    # IEEE Trans. Comm. COM-29(7), Table II. DOI: 10.1109/TCOM.1981.1095108
    # BT=0.5, Eb/N0=10dB → BER ≈ 3.3e-6
    result = gmsk_awgn_ber(ebn0_db=10.0, bt=0.5)
    assert abs(result - 3.3e-6) / 3.3e-6 < 0.15, f"Got {result}"


def test_gmsk_awgn_ber_bt03_at_10db():
    # REF: Murota & Hirade 1981, Table II.
    # BT=0.3, Eb/N0=10dB → BER ≈ 7.5e-6
    result = gmsk_awgn_ber(ebn0_db=10.0, bt=0.3)
    assert abs(result - 7.5e-6) / 7.5e-6 < 0.20, f"Got {result}"


def test_kdist_second_moment():
    # REF: V5 PROJECT_DOCUMENTATION §1; Jao 1984 Eq.(5): E[|h|²] = b²
    result = kdist_second_moment(alpha=10, b=2)
    assert abs(result - 4.0) < 1e-6, f"Got {result}"
