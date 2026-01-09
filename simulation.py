from FECsim import *


def run_experiment():
    sim = FECsim()
    data_len = 100  # dlugosc wiadomosci w bitach
    # generuje losowe dane (zera i jedynki)
    input_data = np.random.randint(0, 2, data_len)

    print(f"Start testu FEC - dlugosc danych: {data_len}")

    # przetwarzanie nadawcze (TX chain)
    # 1. wybielanie danych (scrambling)
    scrambled = scrambler(input_data)
    # 2. kodowanie kanalowe (dodaje nadmiarowosc)
    encoded = sim.encoder(scrambled)
    total_coded_bits = len(encoded)

    print(f"bity po zakodowaniu: {total_coded_bits} (nadmiarowosc ~2x)")
    print("-" * 60)
    print(f"{'liczba bledow':<15} | {'bledy w bitach':<15} | {'stan naprawy':<20}")
    print("-" * 60)

    # testowanie rosnacej ilosci bledow w kanale
    # sprawdzamy od 0 do 15 bledow
    for errors_inserted in range(0, 16):
        # 1. kanal z bledami (symulacja powietrza/zaklocen)
        noisy_encoded = inject_errors(encoded, errors_inserted)

        # 2. dekodowanie (RX chain)
        # najpierw viterbi probuje naprawic bity
        decoded_scrambled = sim.decoder(noisy_encoded)
        # potem odszyfrowanie (descrambling)
        output_data = scrambler(decoded_scrambled)

        # 3. weryfikacja
        # liczy ile bitow rozni sie miedzy oryginalem a tym co odebralismy
        bit_errors_after_decoding = np.sum(input_data != output_data)

        # ocena skutecznosci
        status = "IDEALNIE" if bit_errors_after_decoding == 0 else "DEKODER PADL"

        print(f"{errors_inserted:<15} | {bit_errors_after_decoding:<15} | {status}")

if __name__ == "__main__":
    run_experiment()