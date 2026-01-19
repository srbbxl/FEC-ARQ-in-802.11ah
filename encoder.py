import numpy as np

class ConvolutionalEncoder:
    def __init__(self):
        # parametry standardu
        self.constraint = 7 # K
        # wielomiany generujace
        self.g1 = np.array([1, 0, 1, 1, 0, 1, 1], dtype=int) #oct 133
        self.g2 = np.array([1, 1, 1, 1, 0, 0, 1], dtype=int) #oct 171
        # zerowy stan rejestru
        self.state = np.zeros(self.constraint - 1, dtype=int)

    def encode(self, bits):
        coded_bits = []
        # czyscimy rejestr przy kazdej iteracji
        self.state = np.zeros(self.constraint - 1, dtype=int)
        for bit in bits:

            window = np.concatenate(([bit], self.state))
            out_a = np.sum(window & self.g1) % 2
            out_b = np.sum(window & self.g2) % 2
            coded_bits.append(out_a)
            coded_bits.append(out_b)
            window = window[:-1]
            self.state = window
        return np.array(coded_bits).flatten()
