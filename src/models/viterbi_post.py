"""Viterbi and CRF post-processors for GMSK bit sequence refinement.

Two implementations:
  1. viterbi_refine()  — GMSK trellis Viterbi using model soft probabilities as
                         unary potentials. State = last L bits (GMSK memory).
                         Branch metric is state-independent (no channel model).
  2. crf_refine()      — Learned 2×2 pairwise CRF. Transition matrix fit on val
                         set from model soft outputs + ground truth labels.

Gate both on val BER before applying to test. If neither helps, drop and note.
"""

import math
import numpy as np


# ── GMSK trellis Viterbi ──────────────────────────────────────────────────────

def _gmsk_memory(bt: float) -> int:
    """GMSK ISI memory length in symbols: ceil(1 / BT), capped at 4."""
    return min(int(math.ceil(1.0 / bt)), 4)


def viterbi_refine(
    bit_probs: np.ndarray,
    bt_product: float,
) -> np.ndarray:
    """
    GMSK trellis Viterbi on model soft probabilities.

    bit_probs : (B, T) float in (0, 1) — model per-bit soft outputs
    bt_product: 0.3 or 0.5 (GMSK BT product for this frame batch)

    Returns (B, T) hard decisions {0, 1}.

    NOTE: Branch metrics are state-independent (we lack the raw IQ at this
    stage). This enforces trellis *reachability* but not channel-model
    likelihood — net effect is mild sequence smoothing. Gate on val BER.
    """
    B, T = bit_probs.shape
    L    = _gmsk_memory(bt_product)
    n_states = 1 << L     # 2^L states

    EPS = 1e-7
    log_p1 = np.log(np.clip(bit_probs, EPS, 1 - EPS))       # (B, T)
    log_p0 = np.log(np.clip(1 - bit_probs, EPS, 1 - EPS))   # (B, T)

    # path_metric[b, s] = cumulative log-prob for batch b in state s
    NEG_INF = -1e9
    path_metric = np.full((B, n_states), NEG_INF, dtype=np.float64)
    path_metric[:, 0] = 0.0  # all frames start in state 0 (last L bits = 0)

    # traceback[t, b, s] = (prev_state, bit_emitted)
    traceback = np.zeros((T, B, n_states, 2), dtype=np.int32)

    for t in range(T):
        new_metric = np.full_like(path_metric, NEG_INF)
        for s in range(n_states):
            for b in range(2):
                # next state: shift in bit b from the left
                ns = ((s << 1) | b) & (n_states - 1)
                m  = path_metric[:, s] + (log_p1[:, t] if b else log_p0[:, t])
                better = m > new_metric[:, ns]
                new_metric[better, ns] = m[better]
                traceback[t][better, ns, 0] = s
                traceback[t][better, ns, 1] = b
        path_metric = new_metric

    # Traceback
    best_state = np.argmax(path_metric, axis=1)      # (B,)
    decoded = np.zeros((B, T), dtype=np.int32)
    state = best_state.copy()
    for t in range(T - 1, -1, -1):
        bits  = traceback[t, np.arange(B), state, 1]
        prevs = traceback[t, np.arange(B), state, 0]
        decoded[:, t] = bits
        state = prevs

    return decoded.astype(np.float32)


# ── Learned 2×2 CRF ──────────────────────────────────────────────────────────

class PairwiseCRF:
    """
    1D binary CRF with a single 2×2 transition matrix.
    Unary potentials  : model log-probabilities.
    Pairwise potential: learned log-transition(b_t → b_{t+1}).

    Fit on val set, apply to test.
    """

    def __init__(self):
        self.log_trans = np.zeros((2, 2), dtype=np.float64)   # uniform init

    def fit(self, bit_probs: np.ndarray, labels: np.ndarray):
        """
        Estimate transition matrix from val set.
        bit_probs: (N, T) soft predictions
        labels   : (N, T) ground-truth bits {0, 1}
        """
        trans = np.zeros((2, 2), dtype=np.float64)
        for t in range(labels.shape[1] - 1):
            for b0 in range(2):
                for b1 in range(2):
                    trans[b0, b1] += np.sum(
                        (labels[:, t] == b0) & (labels[:, t + 1] == b1)
                    )
        trans /= trans.sum(axis=1, keepdims=True) + 1e-8
        self.log_trans = np.log(np.clip(trans, 1e-8, 1.0))

    def decode(self, bit_probs: np.ndarray) -> np.ndarray:
        """Viterbi with learned transitions. Returns (B, T) hard decisions."""
        B, T  = bit_probs.shape
        EPS   = 1e-7
        log_p = np.stack([
            np.log(np.clip(1 - bit_probs, EPS, 1 - EPS)),
            np.log(np.clip(bit_probs,     EPS, 1 - EPS)),
        ], axis=2)          # (B, T, 2)

        NEG_INF = -1e9
        dp      = np.full((B, 2), NEG_INF)
        dp[:, 0] = log_p[:, 0, 0]
        dp[:, 1] = log_p[:, 0, 1]

        back = np.zeros((T, B, 2), dtype=np.int32)
        for t in range(1, T):
            new_dp = np.full_like(dp, NEG_INF)
            for b in range(2):
                cand0 = dp[:, 0] + self.log_trans[0, b] + log_p[:, t, b]
                cand1 = dp[:, 1] + self.log_trans[1, b] + log_p[:, t, b]
                take0 = cand0 >= cand1
                new_dp[:, b] = np.where(take0, cand0, cand1)
                back[t, :, b] = np.where(take0, 0, 1)
            dp = new_dp

        decoded = np.zeros((B, T), dtype=np.int32)
        state = np.argmax(dp, axis=1)
        decoded[:, T - 1] = state
        for t in range(T - 2, -1, -1):
            state = back[t + 1, np.arange(B), state]
            decoded[:, t] = state

        return decoded.astype(np.float32)
