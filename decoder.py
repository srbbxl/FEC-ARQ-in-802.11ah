import numpy as np

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