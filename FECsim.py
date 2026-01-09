import numpy as np


# symulacja powstawania bledow przy transmisji
def inject_errors(bits, error_count):
    # robi kopie tablicy zeby nie psuc oryginalnych danych w pamieci
    corrupted = bits.copy()

    # losuje unikalne indeksy (miejsca w tablicy), ktore zostana uszkodzone
    # replace=False gwarantuje ze nie wylosujemy dwa razy tego samego indeksu
    indices = np.random.choice(len(bits), error_count, replace=False)

    for idx in indices:
        # odwracanie bitu (bit flip): jak jest 0 to robi 1, jak 1 to robi 0
        corrupted[idx] = 1 - corrupted[idx]
    return corrupted


# mieszanie bitow w celu unikniecia dlugich ciagow zer i jedynek - dziala w dwie strony (kodowanie i odkodowanie)
def scrambler(bits, seed=0b1011101):
    # scrambler z wielomianem x^7 + x^4 + 1
    # wykorzystuje rejestr przesuwny (LFSR)
    state = seed
    output = np.zeros_like(bits)

    for i in range(len(bits)):
        # obliczamy bit sprzezenia zwrotnego (feedback) na podstawie wielomianu
        # state >> 6 wyciaga 7. bit (reprezentuje x^7)
        # state >> 3 wyciaga 4. bit (reprezentuje x^4)
        # ^ to operacja XOR (dodawanie modulo 2)
        # & 1 maskuje wynik zeby miec pewnosc ze to pojedynczy bit (0 lub 1)
        feedback = ((state >> 6) ^ (state >> 3)) & 1

        # wlasciwe szyfrowanie: xorujemy bit danych z wyliczonym feedbackiem
        output[i] = bits[i] ^ feedback

        # aktualizacja stanu rejestru na nastepny krok
        # state << 1 przesuwa caly rejestr w lewo (robi miejsce na nowy bit)
        # | feedback wstawia obliczony wczesniej bit na najmlodsza pozycje
        # & 0x7F (binarnie 1111111) ucina wszystko powyzej 7 bitow, zeby rejestr sie nie rozrosl
        state = ((state << 1) | feedback) & 0x7F
    return output


class FECsim:
    def __init__(self):
        # parametry 802.11ah: K=7 (dlugosc ograniczenia), R=1/2 (poziom nadmiarowosci)
        # te liczby definiuja polaczenia wewnatrz kodera splotowego
        self.poly_a = 0o133  # 1011011 binarnie
        self.poly_b = 0o171  # 1111001 binarnie

        # prekalkulacja kratownicy dla dekodera, w sumie to taka mapa
        # po ktorej dekoder sie rusza
        # next_state - jak jestes w stanie X i przyjdzie bit Y, to gdzie idziesz?
        self.trellis_next_state = np.zeros((64, 2), dtype=int)
        # output - jestes w stanie X i przyjdzie bit Y, to co koder wypluwa?
        self.trellis_output = np.zeros((64, 2), dtype=int)

        # odpala budowanie mapy raz, zeby potem nie liczyc tego w kolko
        self._build_trellis()

    def _build_trellis(self):
        # buduje mape przejsc stanow
        # iteruje po wszystkich mozliwych stanach rejestru (2^6 = 64 stany)
        for state in range(64):
            # dla kazdego mozliwego wejscia (bit 0 albo 1)
            for bit in [0, 1]:
                # symulacja przesuniecia rejestru w koderze
                # w lewo, dodanie bita wejsciowego, maska do 6 bitow
                next_state = ((state << 1) | bit) & 0x3F

                # tworzy tymczasowy "pelny" rejestr (7 bitow) zeby policzyc wyjscie
                full_reg = (state << 1) | bit

                # obliczanie parzystosci dla wielomianu A
                # AND z maska wielomianu wybiera odpowiednie bity
                # count('1') zlicza jedynki
                # % 2 daje wynik modulo 2 (czyli parzystosc)
                out_a = bin(full_reg & self.poly_a).count('1') % 2

                # to samo dla wielomianu B
                out_b = bin(full_reg & self.poly_b).count('1') % 2

                # zapisuje wynik do tablic lookup table
                self.trellis_next_state[state][bit] = next_state
                # skleja dwa bity wyjsciowe w jedna liczbe (np 1 i 0 -> 10 binarnie -> 2 decymalnie)
                self.trellis_output[state][bit] = (out_a << 1) | out_b

    def encoder(self, bits):
        # koder splotowy
        state = 0
        encoded = []
        # dodaje 6 zer na koniec (tail bits) zeby domknac kodowanie poprawnie
        # to resetuje rejestr kodera do stanu 0 na koniec transmisji
        # kluczowe dla dekodera zeby wiedzial gdzie skonczyc
        padded_bits = np.concatenate((bits, [0] * 6))

        for bit in padded_bits:
            # wsuwa bit do rejestru
            full_reg = (state << 1) | bit

            # liczy bity wyjsciowe tak samo jak przy budowaniu trellis
            out_a = bin(full_reg & self.poly_a).count('1') % 2
            out_b = bin(full_reg & self.poly_b).count('1') % 2

            encoded.extend([out_a, out_b])

            # aktualizuje stan pamieci (zostawia tylko 6 ostatnich bitow)
            state = full_reg & 0x3F
        return np.array(encoded, dtype=int)

    def decoder(self, encoded_bits):
        """
        dekoder viterbiego
        logika: szuka sciezki w kratownicy ktora najmniej rozni sie
        od tego co odebralismy (minimalizuje bledy)
        """
        # liczba krokow czasowych (ile par bitow odebralismy)
        n_steps = len(encoded_bits) // 2

        # inicjalizacja kosztow sciezek - na start same nieskonczonosci (duza liczba)
        path_metrics = np.full(64, 1e6)
        # wiemy ze zaczynamy w stanie 0, wiec koszt dojscia tam to 0
        path_metrics[0] = 0

        # tablica pamieci "okruszkow" - pamieta skad przyszlismy do danego stanu
        traceback = np.zeros((n_steps, 64), dtype=int)

        # faza 1 - liczenie kosztow
        for t in range(n_steps):
            # pobiera pare odebranych bitow dla aktualnego kroku
            # skleja je w jedna liczbe (0-3)
            rx_pair = (encoded_bits[2 * t] << 1) | encoded_bits[2 * t + 1]

            # tablica na koszty w kolejnym kroku
            new_metrics = np.full(64, 1e6)

            for state in range(64):
                # optymalizacja: jak stan jest nieosiagalny (koszt inf), to go pomijamy
                if path_metrics[state] >= 1e6: continue

                # sprawdza obie mozliwe sciezki (gdyby nadano 0 lub 1)
                for bit in [0, 1]:
                    # patrzy w mape trellis gdzie wyladujemy i co powinnismy odebrac
                    next_s = self.trellis_next_state[state][bit]
                    expected_out = self.trellis_output[state][bit]

                    # ile bitow sie rozni (odleglosc hamminga)
                    # to jest serce naprawiania bledow
                    # XOR (^) zwraca 1 tam gdzie bity sa rozne
                    # count('1') zlicza te roznice (czyli bledy)
                    cost = bin(rx_pair ^ expected_out).count('1')

                    # ACS (Add-Compare-Select)
                    # sprawdza czy ta nowa sciezka jest tansza (mniej bledow) niz ta ktora juz znamy
                    if path_metrics[state] + cost < new_metrics[next_s]:
                        # jak tak, to aktualizuje koszt
                        new_metrics[next_s] = path_metrics[state] + cost
                        # i zapamietuje w traceback ze przyszlismy ze 'state'
                        traceback[t][next_s] = state

            # zamienia tablice metryk na nowa dla nastepnego kroku
            path_metrics = new_metrics

        # faza 2 - odzyskiwanie danych
        decoded = []
        state = 0  # wiemy ze konczymy na stanie 0 przez tail bits

        # idzie od ostatniego kroku do pierwszego (-1)
        for t in range(n_steps - 1, -1, -1):
            # patrzy w tablice pamieci skad sie tu wzielismy
            prev = traceback[t][state]

            # dedukcja jaki bit musial byc na wejsciu zeby przejsc z 'prev' do 'state'
            # symuluje przejscie z 'prev' przy bicie 0
            candidate_state_0 = ((prev << 1) | 0) & 0x3F

            # jak pasuje do aktualnego stanu, to znaczy ze bit byl 0, inaczej 1
            bit = 0 if candidate_state_0 == state else 1

            decoded.append(bit)
            # cofa sie do poprzedniego stanu
            state = prev

        # lista jest od tylu, wiec trzeba ja odwrocic
        # i uciac 6 ostatnich bitow, bo to tylko tail bits (smieci techniczne)
        return np.array(decoded[::-1][:-6])