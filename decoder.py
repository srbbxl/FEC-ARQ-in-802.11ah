import numpy as np

def int_to_bits(num, length):
    binary_string = np.binary_repr(num, width=length)
    return np.array([int(x) for x in binary_string], dtype=int)

class Decoder:
    def __init__(self, encoder_instance):
        self.encoder = encoder_instance
        # ilosc mozliwych stanow (2^6 = 64)
        self.num_states = 2 ** (encoder_instance.constraint - 1)

        # tabela przejść [stan_obecny][wejscie] -> stan_nastepny
        self.next_state_table = np.zeros((self.num_states, 2), dtype=int)

        # tabela wyjsc [stan_obecny][wejscie] -> [bit_a][bit_b]
        # dwa razy dluzsze, bo rate 1/2
        self.output_table = np.zeros((self.num_states, 2, 2), dtype=int)

        # bity wejsciowe
        self.input_bit_table = np.zeros((self.num_states, self.num_states), dtype=int)

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

    def decode(self, received_bits):
        # sprawdzamy długośc, musi być podzielna przez 2, bo rate 1/2
        if received_bits % 2 != 0:
            received_bits = received_bits[:-1]

        # ilosc krokow czasowych
        n_steps = len(received_bits) // 2

        # koszty (metryki) - stan zero ma koszt 0, reszta nieskonczonosc
        path_metrics = np.full(self.num_states, 99999999.9)
        path_metrics[0] = 0

        # trellis (kratownica) - tu zapisujemy decyzje 0 albo 1
        # tu zapisujemy POPRZEDNI STAN (skąd przybyliśmy), a nie bit wejściowy
        trellis = np.zeros((n_steps, self.num_states), dtype=int)

        # pętla po krokach
        for x in range(n_steps):
            # pobieramy parę bitów z odebranego sygnału
            received_pair = received_bits[x * 2:x * 2 + 2]

            # nowe koszty (metryki) na ten krok
            new_path_metrics = np.full(self.num_states, 99999999.9)

            # pętla po stanach
            for prev_state in range(self.num_states):
                # jeśli koszt dotarcia jest nieskończony, olewamy stan
                if path_metrics[prev_state] >= 99999999.9:
                    continue

                # sprawdzamy dwie gałęzie (bity 0 i 1)
                for input_bit in [0, 1]:
                    # destynacja
                    next_state = self.next_state_table[prev_state][input_bit]

                    # oczekiwanie
                    expected_output = self.output_table[prev_state][input_bit]

                    # obliczamy koszt (odległość Hamminga), porównujemy z odebraną parą oczekiwaną
                    # XOR tam gdzie jest różnica, suma to ilość różnic
                    branch_metric = np.sum(received_pair ^ expected_output)

                    # całkowity koszt ścieżki
                    new_metric = path_metrics[prev_state] + branch_metric

                    # czy znaleziona ścieżka jest lepsza:
                    if new_metric < new_path_metrics[next_state]:
                        new_path_metrics[next_state] = new_metric

                        #zapisujemy STAN w historii jeśli był lepszy, natomiast bit odzyskamy później
                        trellis[x][next_state] = prev_state

            path_metrics = new_path_metrics

        # dekodowany sygnał
        decoded_bits = []

        # bierzemy stan z najmniejszym błędem na końcu
        current_state = np.argmin(path_metrics)

        # traceback od końca do początku (szukaj swego wątku) (pw hnl reference)
        for x in range(n_steps - 1, -1, -1):
            # patrzymy w trellis, stamtąd przeszliśmy do current_state w kroku x
            prev_state = trellis[x][current_state]

            # bit, który przeniósł nas z prev do current
            bit = self.input_bit_table[prev_state][current_state]
            decoded_bits.append(bit)

            # cofamy się
            current_state = prev_state

        # odwracamy listę dekodowanego sygnału
        return np.array(decoded_bits[::-1])