# macierze - generowanie losowych sygnalow, szumu, operacji wektorowych
import numpy as np
# commpy - biblioteka telekomunikacyjna - gotowe funkcje do kodowania splotowego i algorytmu viterbiego
import commpy.channelcoding.convcode as cc
import matplotlib.pyplot as plt
# funkcja błędu - funkcja błędu (complementary error func) - potrzebna tylko do narysowania idealnego wzorca, żeby
# sprawdzić czy symulacja działa poprawnie
from scipy.special import erfc


class WifiBCC:
    def __init__(self):
        # pamięć rejestru przesuwnego K
        self.memory = np.array([6])
        # wielomiany generacyjne (0o to zapis ósemkowy)
        self.g_matrix = np.array([[0o133, 0o171]])
        # krata (trellis) do algorytmu Viterbiego
        self.trellis = cc.Trellis(self.memory, self.g_matrix)

    def encode(self, bits):
        # koduje bity na podstawie wyżej zdefiniowanej kraty
        return cc.conv_encode(bits, self.trellis)

    def decode(self, coded_bits):
        # dekodowanie Viterbiego (twarde zera i jedynki, a nie wartości prawdopodobieństwa)
        return cc.viterbi_decode(coded_bits, self.trellis, decoding_type='hard')


def run_simulation(snr_range_db, num_bits=10000):
    ber_coded = []
    ber_uncoded = []

    bcc = WifiBCC()
    code_rate = 1 / 2

    print(f"rozpoczynam symulację dla {num_bits} bitów...")

    for snr_db in snr_range_db:
        # generacja danych
        tx_bits = np.random.randint(0, 2, num_bits)

        # ŚCIEŻKA KODOWANA (BCC)
        # dodajemy 'flush bits' (zera), aby zresetować rejest r na końcu
        # pozwala to Viterbiemu poprawnie zdekodować ostatnie bity
        tx_bits_flushed = np.append(tx_bits, np.zeros(bcc.memory[0], dtype=int))

        # kodowanie (podwaja ilosc bitow)
        encoded_bits = bcc.encode(tx_bits_flushed)

        # modulacja BPSK (0 -> -1, 1 -> 1)
        symbols_coded = 2 * encoded_bits - 1

        # obliczenie mocy szumu dla systemu kodowanego
        # Eb/N0 = SNR_dB. Musimy uwzględnić Rate (R)
        # Sigma = sqrt(1 / (2 * R * 10^(SNR/10)))
        noise_power_coded = 1.0 / (2 * code_rate * 10 ** (snr_db / 10.0))
        # generujemy szum gaussowski mający na celu symulować przepływ przez kabel/powietrze
        noise_coded = np.sqrt(noise_power_coded) * np.random.randn(len(symbols_coded))
        rx_signal_coded = symbols_coded + noise_coded
        # demodulacja, jeśli wartość odebrana jest większa niż 0 to uznajemy za 1, jeśli nie to za 0
        rx_bits_coded = (rx_signal_coded > 0).astype(int)

        # dekodowanie - Viterbi
        decoded_bits_flushed = bcc.decode(rx_bits_coded)

        # usuwamy bity flushujące i liczymy błędy
        decoded_bits = decoded_bits_flushed[:len(tx_bits)]

        # liczymy bit error rate (w tablicy True oznacza błąd)
        ber_c = np.mean(tx_bits != decoded_bits)
        ber_coded.append(ber_c)

        # ŚCIEŻKA NIEKODOWANA (Referencja)
        # dla porównania zwykły BPSK bez kodowania
        symbols_uncoded = 2 * tx_bits - 1

        # dla uncoded R = 1, więc szum jest mniejszy przy tym samym Eb/N0
        noise_power_uncoded = 1.0 / (2 * 1.0 * 10 ** (snr_db / 10.0))
        noise_uncoded = np.sqrt(noise_power_uncoded) * np.random.randn(len(symbols_uncoded))

        rx_signal_uncoded = symbols_uncoded + noise_uncoded
        rx_bits_uncoded = (rx_signal_uncoded > 0).astype(int)

        ber_u = np.mean(tx_bits != rx_bits_uncoded)
        ber_uncoded.append(ber_u)

    return ber_coded, ber_uncoded


# parametry symulacji
snr_range = np.arange(0, 11, 1)  # od 0 do 10 dB
ber_coded, ber_uncoded = run_simulation(snr_range, num_bits=50000)

# teoretyczny wykres BPSK dla weryfikacji (linia przerywana)
theory_ber = 0.5 * erfc(np.sqrt(10 ** (snr_range / 10.0)))

# rysowanie wykresu
plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, ber_uncoded, 'bo-', label='Symulacja: Bez kodowania (BPSK)')
plt.semilogy(snr_range, theory_ber, 'b--', alpha=0.5, label='Teoria: BPSK')
plt.semilogy(snr_range, ber_coded, 'rs-', label='Symulacja: WiFi BCC (K=7, R=1/2)')

plt.title('Porównanie BER: BPSK vs BPSK + Kodowanie Splotowe (WiFi)')
plt.xlabel('Eb/N0 [dB]')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()