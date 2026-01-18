import numpy as np
from numpy import dtype

from encoder import ConvolutionalEncoder

def int_to_bits(num, length):
    binary_string = np.binary_repr(num, width=length)
    return np.array([int(x) for x in binary_string], dtype=int)

class Decoder:
    def __init__(self, encoder):
        self.encoder = encoder
        # ilosc mozliwych stanow (2^6 = 64)
        self.num_states = 2 ** (encoder.constraint - 1)

        # tabela przejść [stan_obecny][wejscie] -> stan_nastepny
        self.next_state_table = np.zeros((self.num_states, 2), dtype=int)

        # tabela wyjsc [stan_obecny][wejscie] -> [bit_a][bit_b]
        # dwa razy dluzsze, bo rate 1/2
        self.output_table = np.zeros((self.num_states, 2, 2), dtype=int)

        # mapa mozliwych przejsc
        for state in range(self.num_states):
            # przygotowanie stanu kodera. zamiana "state" na bity
            bit_state = int_to_bits(state, self.encoder.constraint - 1)

            # pętla dla wejść 0 i 1
            for input_bit in [0, 1]:
                #ustawianie stanu kodera na sztywno
                self.encoder.state = bit_state.copy()
                #używanie metody encode() z pliku kodera, żeby zakodować JEDEN bit wejściowy
                encoded_output = self.encoder.encode([input_bit])
                # zapis wyniku w tabeli
                self.output_table[state][input_bit] = encoded_output
                # stan nowego kodera
                # zamieniamy to z powrotem na int, żeby zaipsać w tabeli
                new_state_bits = self.encoder.state
                new_state_int = int("".join(str(x) for x in new_state_bits), 2)
                self.next_state_table[state][input_bit] = new_state_int

    def decode(self, recieved_bits):
        # sprawdzamy długośc, musi być podzielna przez 2, bo rate 1/2
        if recieved_bits % 2 != 0:
            recieved_bits = recieved_bits[:-1]

        # ilosc krokow czasowych
        n_steps = len(recieved_bits) // 2

        # koszty (metryki) - stan zero ma koszt 0, reszta nieskonczonosc
        path_metrics = np.full(self.num_states, 99999999.9)
        path_metrics[0] = 0

        # trellis (kratownica) - tu zapisujemy decyzje 0 albo 1
        # wymiar: [czas][stan] -> bit wejściowy, ktory nas tutaj doprowadził
        trellis = np.zeros((n_steps, self.num_states), dtype=int)

        # pętla po krokach
        for x in range(n_steps):
            # pobieramy parę bitów z odebranego sygnału
            recieved_pair = recieved_bits[x * 2:x * 2 + 2]

            # nowe koszty (metryki) na ten krok
            new_path_metrics = np.full(self.num_states, 99999999.9)

            # pętla po możliwych stanach
            for state in range(self.num_states):
                pass

            path_metrics = new_path_metrics


print(np.zeros((3, 64), dtype=int))