import numpy as np


class ConvolutionalEncoder:
    def __init__(self):
        # parametry (K=7)
        self.constraint = 7
        # wielomiany generujace (133 i 171)
        self.g1 = np.array([1, 0, 1, 1, 0, 1, 1], dtype=int)
        self.g2 = np.array([1, 1, 1, 1, 0, 0, 1], dtype=int)
        # na start czysto, same zera w pamieci
        self.state = np.zeros(self.constraint - 1, dtype=int)

    def reset(self):
        # czyscimy pamiec do zera
        self.state = np.zeros(self.constraint - 1, dtype=int)

    def encode(self, bits):
        coded_bits = []

        # lecimy po kazdym bicie z wejscia
        for bit in bits:
            # tworzymy okno: [nowy_bit, stary_stan...]
            window = np.concatenate(([bit], self.state))

            # liczymy XORy (suma modulo 2) dla wielomianow
            out_a = np.sum(window & self.g1) % 2
            out_b = np.sum(window & self.g2) % 2

            # dopisujemy do wyniku
            coded_bits.append(out_a)
            coded_bits.append(out_b)

            # przesuwamy tasme - wyrzucamy ostatni bit, reszta zostaje jako stan
            window = window[:-1]
            self.state = window

        return np.array(coded_bits).flatten()