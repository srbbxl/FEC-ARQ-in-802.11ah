import numpy as np
from encoder import ConvolutionalEncoder
from decoder import Decoder


def apply_channel_noise(encoded_bits, ber):
    # losujemy wartosci 0-1 dla kazdego bitu
    random_vals = np.random.rand(len(encoded_bits))
    # maska bledow: tam gdzie wylosowalo mniej niz BER, tam jebnie błąd
    error_mask = (random_vals < ber).astype(int)
    # psujemy sygnał XORem
    corrupted = encoded_bits ^ error_mask
    return corrupted


if __name__ == "__main__":
    # 1. Odpalamy maszyny
    enc = ConvolutionalEncoder()
    dec = Decoder(enc)

    # 2. Generujemy losowe dane (np. 100 bitów)
    input_data = np.random.randint(0, 2, 100)
    print(f"Dane wejsciowe: {input_data[:15]}... (dlugosc: {len(input_data)})")

    # WAŻNE: Dodajemy "Tail Bits" (Padding).
    # Musimy dorzucic 6 zer (K-1), zeby Viterbi mogl ladnie zakonczyc prace na stanie 0.
    # Jak tego nie zrobisz, ostatnie bity beda z dupy.
    padding = np.zeros(enc.constraint - 1, dtype=int)
    data_with_padding = np.concatenate((input_data, padding))

    # kodowanie
    enc.reset()  # dla pewnosci czyscimy
    encoded_signal = enc.encode(data_with_padding)
    print(f"Zakodowane: {encoded_signal[:20]}... (dlugosc: {len(encoded_signal)})")

    # wprowadzamy błędy
    BER = 0.05  # 5% szans ze cos nie zadziała
    received_signal = apply_channel_noise(encoded_signal, BER)

    # zlicza ile bledow wpadlo
    errors = np.sum(encoded_signal != received_signal)
    print(f"kanal dorzucil {errors} bledow")

    # dekodowanie viterbiego
    decoded_full = dec.decode(received_signal)

    # odcinamy 6 zer, ktore byly dodane do wyczyszczenia rejestru
    decoded_data = decoded_full[:len(input_data)]

    print(f"Zdekodowane: {decoded_data[:15]}...")

    # sprawdzanie poprawności długości sygnału
    bit_errors = np.sum(input_data != decoded_data)

    print("-" * 30)
    if bit_errors == 0:
        print(f"SUKCES - wsyzstko odzyskane")
    else:
        print(f"Dekodowanie w pełni nie udało się - postalo {bit_errors} bledow")