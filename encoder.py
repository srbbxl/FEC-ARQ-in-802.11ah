import numpy as np

class Encoder:
    def __init__(self, bits):
        # parametry standardu
        self.constraint = 7 # K
        # wielomiany generujace
        self.g1 = np.array([1, 0, 1, 1, 0, 1, 1], dtype=int) #oct 133
        self.g2 = np.array([1, 1, 1, 1, 0, 0, 1], dtype=int) #oct 171
        # zerowy stan rejestru
        self.state = np.zeros(self.constraint - 1, dtype=int)
        # wejsciowy ciag danych
        self.bits = bits


    def encode(self):
        coded_bits = []
        # czyscimy rejestr przy kazdej iteracji
        self.state = np.zeros(self.constraint - 1, dtype=int)
        for bit in self.bits:

            window = np.concatenate(([bit], self.state))
            out_a = np.sum(window & self.g1) % 2
            out_b = np.sum(window & self.g2) % 2
            coded_bits.append(out_a)
            coded_bits.append(out_b)
            window = window[:-1]
            self.state = window
        return np.array(coded_bits).flatten()

    def apply_noise(self, ber):
        random_vals = np.random.rand(len(self.bits))
        print(random_vals)
        noise = (random_vals < ber).astype(int)
        print(noise)
        corrupted_bits = self.bits ^ noise
        return corrupted_bits

encoder = Encoder([1, 0, 1, 0, 1, 1])
print(encoder.apply_noise(1/2))