# References

---

## Papers Directly Used in This Project

[1] B. Zhu et al., "GMSK Demodulation Combining 1D-CNN and Bi-LSTM Network Over Strong Solar Wind Turbulence," *Radio Science*, 2023. Dataset: Zenodo.
*(Primary baseline — the paper and dataset this project is built on.)*

[2] K. Murota and K. Hirade, "GMSK Modulation for Digital Mobile Radio Telephony," *IEEE Transactions on Communications*, vol. COM-29, no. 7, pp. 1044–1050, Jul. 1981. DOI: 10.1109/TCOM.1981.1095108.
*(GMSK BER theory and BT-product efficiency factors used in the signal simulator.)*

[3] J. K. Jao, "Amplitude Distribution of Composite Terrain Scattered and Line-of-Sight Signal and Its Application to Land Mobile Radio Channels," *IEEE Transactions on Antennas and Propagation*, vol. 32, no. 10, pp. 1049–1062, Oct. 1984. DOI: 10.1109/TAP.1984.1143233.
*(K-distribution envelope PDF and moment equations.)*

[4] K. D. Ward, R. J. A. Tough, and S. Watts, *Sea Clutter: Scattering, the K Distribution and Radar Performance*, IET, 2006, Ch. 4, Eq. (4.19)–(4.20).
*(Complex K-distribution formulation adapted for ionospheric channel simulation.)*

[5] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752, Dec. 2023.
*(Original Mamba (SSM) architecture — foundation for all Mamba variants used.)*

[6] T. Dao and A. Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," arXiv:2405.21060, May 2024.
*(Mamba-2 — the SSM layer used in BiMamba2 and MambaNet blocks.)*

[7] A. Lahoti, K. Y. Li, B. Chen, C. Wang, A. Bick, J. Z. Kolter, T. Dao, and A. Gu, "Mamba-3: Improved Sequence Modeling using State Space Principles," arXiv:2503.15569, Mar. 2026.
*(Mamba-3 — the higher-order SSM variant built from source for the V5 model.)*

[8] D. Luan, C. Liang, J. Huang, Z. Lin, K. Meng, J. Thompson, and C.-X. Wang, "MambaNet: Mamba-Assisted Channel Estimation Neural Network with Attention Mechanism," arXiv:2601.17108, Jan. 2026. To appear at ICASSP 2026.
*(Architectural blueprint for our MHA → BiMamba2 model.)*

---

## Additional Papers Referenced / Background Reading

[9] L. Cai, G. Xu, Q. Zhang, Z. Song, and W. Zhang, "Deep Learning Based Channel Estimation for Deep-Space Communications," *IEEE Transactions on Vehicular Technology*, vol. 74, no. 12, pp. 19743–, Dec. 2025.
*(Deep-space channel estimation context — solar scintillation and Doppler over deep-space links.)*

[10] S. Gao, W. Guo, H. Shi, and R. Peng, "IQUMamba-1D: A Mamba-Enhanced 1D U-Net for Single-Channel Communication Signal Blind Source Separation," *Journal of King Saud University – Computer and Information Sciences*, vol. 38, no. 63, Jan. 2026. DOI: 10.1007/s44443-025-00440-5.
*(Mamba applied to raw I/Q signal processing — closely related to our I/Q demodulation approach.)*

[11] R. Zhang et al., "Mamba for Wireless Communications and Networking: Principles and Opportunities," arXiv:2508.00403, Aug. 2025.
*(Survey of Mamba architectures in wireless communications — broader context for SSM-based demodulation.)*

[12] Consultative Committee for Space Data Systems (CCSDS), *Radio Frequency and Modulation Systems — Part 1: Earth Stations and Spacecraft*, CCSDS 401.0-B-32, Oct. 2021.
*(Space communications standards — modulation and RF system specifications.)*
