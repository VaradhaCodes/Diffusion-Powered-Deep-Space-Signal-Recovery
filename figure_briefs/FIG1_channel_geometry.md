# Figure — Channel Geometry Diagram

## What this diagram is

Two-panel conceptual diagram (not a data chart). Pure illustration. Shows:
- (a) The Solar–Earth–Probe geometry with real spacecraft positions
- (b) The signal processing chain from transmitter to receiver

---

## Panel (a) — SEP Geometry (Space diagram)

**What to draw:**
A schematic top-down view of the solar system (not to scale — schematic only).

**Objects to include:**
1. **Sun** — center, large circle, yellow/orange
2. **Earth** — medium circle, left of center or bottom, blue-green
3. **Voyager 1** — tiny dot, very far from Sun (draw it at far edge with an arrow saying "~173 AU" or "~1 light-day" — November 2026 position)
4. **Mars / DSOC (Psyche)** — small dot between Earth and outer system
5. **Juno (Jupiter)** — slightly farther out than Mars

**Lines/arcs to draw:**
- Signal path line: Voyager 1 → Earth (dashed, to show it crosses near the Sun when SEP is small)
- Label the angle at Earth: "SEP angle (Solar–Earth–Probe)"
- When SEP is small = line passes close to Sun = heavy scintillation (mark this clearly)
- When SEP is large = line clear of Sun = mild scintillation

**Labels:**
- "SEP angle → 0°: signal passes through solar corona → severe K-dist scintillation"
- "SEP angle > 50°: signal avoids corona → near-AWGN propagation"
- Voyager: "Voyager 1, Nov 2026, ~173 AU, signal delay: ~24 hours"
- DSOC: "Psyche DSOC (laser comms)"
- Juno: "Juno, Jupiter orbit"

**Tone:** scientific but clean — this is a diagram not an accurate map. Relative sizes and positions are illustrative.

---

## Panel (b) — Signal Chain Block Diagram

A left-to-right processing pipeline diagram showing what happens to the signal:

```
[Transmitter]
    Spacecraft (e.g. Voyager 1)
    Transmit power: 23W
    Antenna: 3.7m dish
    Frequency: 8.4 GHz (X-band)
         ↓
[GMSK Modulator]
    100-bit frame per burst
    8 samples/symbol
    BT ∈ {0.3, 0.5}
    (Gaussian Minimum Shift Keying)
         ↓
[K-Distribution Fading Channel]
    Amplitude: |h(t)|² ~ Gamma × Exponential
    Scintillation index m ∈ {1.2, 1.4}
    Random phase per frame
    Models solar plasma scintillation
         ↓
[AWGN]
    n(t) ~ CN(0, σ²)
    SNR: -4 to +8 dB
    Free-space path loss: ~315 dB at 173 AU
         ↓
[Received Signal r(t) = √p · h(t) · x(t) + n(t)]
    Shape: (2, 800) — I/Q channels
         ↓
[MambaNet-2ch Receiver]  ← THIS IS WHAT WE BUILT
    CNN Stem
    MHA Attention
    BiMamba2
    FiLM(SNR)
         ↓
[Bit Decisions]
    (100,) bits
    BER measured vs ground-truth
```

**Callout boxes of interest:**
- On the channel box: "K-distribution: models charged-particle scintillation in solar wind"
- On AWGN: "Path loss 315 dB at 173 AU — received power ~10⁻²⁶ W"
- On the receiver box: "Our contribution — beats Zhu 2023 by 27.1%"

---

## Style notes

- This is conceptual art, not a data figure
- Keep it clean and not too detailed — this is meant to orient the reader quickly
- If it's slides: dark background, glowing lines, space aesthetic
- If it's paper: white background, clean boxes, minimal color

Equation reference (for caption or labels):
```
r(t) = √p · h(t) · x(t) + n(t)

|h(t)|² ~ Gamma(α, 1) × Exponential(1/b)
α = 2/(m−1),  b = 2,  m ∈ {1.2, 1.4}
```
