import numpy as np
import zlib

class WiFiHaLow:
    def __init__(self):
        # parameters from 802.11ah documentation
        self.K = 7 # constraint lenght
        self.memory_len = self.K - 1 # registry memory

        # generating polynomials
        self.g1 = 0o133 # 0b1011011
        self.g2 = 0o171 # 0b1111001

        # state trellis for Viterbi algorithm (precalculated)
        self.trellis_next_state = np.zeros((64, 2), dtype=int)
        self.trellis_output = np.zeros((64, 2), dtype=int)
        self._build_trellis()