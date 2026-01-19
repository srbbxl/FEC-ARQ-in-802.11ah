import numpy as np
from encoder import ConvolutionalEncoder
from decoder import Decoder


def apply_channel_noise(encoded_bits, ber):
    """
    Symuluje kanał szumów (BSC).
    Zamienia bity (0->1, 1->0) z prawdopodobieństwem BER.
    """
    # Losujemy liczby 0-1 dla każdego bitu
    random_vals = np.random.rand(len(encoded_bits))
    # Tworzymy maskę błędów (gdzie < BER, tam błąd)
    error_mask = (random_vals < ber).astype(int)
    # Aplikujemy błędy (XOR)
    corrupted = encoded_bits ^ error_mask
    return corrupted


if __name__ == "__main__":
    # 1. Konfiguracja
    enc = ConvolutionalEncoder()
    dec = Decoder(enc)  # Przekazujemy instancję enkodera!

    # 2. Dane wejściowe (np. 10 bitów)
    input_data = np.random.randint(0, 2, 100)
    print(f"Dane wejściowe (długość {len(input_data)}): {input_data[:10]}...")

    # 3. Kodowanie
    encoded_signal = enc.encode(input_data)
    print(f"Zakodowane (długość {len(encoded_signal)}): {encoded_signal[:20]}...")

    # 4. Kanał (Szum)
    BER = 0.05  # 5% szans na błąd
    received_signal = apply_channel_noise(encoded_signal, BER)

    # Policz ile błędów wpadło
    errors = np.sum(encoded_signal != received_signal)
    print(f"Szum w kanale: wprowadzono {errors} błędów.")

    # 5. Dekodowanie
    decoded_data = dec.decode(received_signal)
    print(f"Zdekodowane: {decoded_data[:10]}...")

    # 6. Weryfikacja
    # Uwaga: Viterbi może uciąć ostatnie bity jeśli nie ma flushingu,
    # więc porównujemy tyle, ile odzyskaliśmy.
    length = min(len(input_data), len(decoded_data))
    bit_errors = np.sum(input_data[:length] != decoded_data[:length])

    print("-" * 30)
    if bit_errors == 0:
        print("SUKCES! Wiadomość odzyskana bezbłędnie.")
    else:
        print(f"PORAŻKA. Pozostało błędów: {bit_errors}")