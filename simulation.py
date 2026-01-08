from FECsim import *

def run_experiment():
    sim = FECsim()
    data_len = 5000  # message length in bits
    input_data = np.random.randint(0, 2, data_len)

    print(f"FEC test start - data length: {data_len}")

    # processing tx
    scrambled = scrambler(input_data)
    encoded = sim.encoder(scrambled)
    total_coded_bits = len(encoded)

    print(f"bits after encoding: {total_coded_bits} (redundancy ~2x)")
    print("-" * 60)
    print(f"{'errors cnt':<15} | {'bit errors':<15} | {'repair status':<20}")
    print("-" * 60)

    # testing rising number of errors in channel
    # checking from 0 to 15 errors
    for errors_inserted in range(0, 16):
        # 1. channel with errors
        noisy_encoded = inject_errors(encoded, errors_inserted)

        # 2. decoding
        decoded_scrambled = sim.decoder(noisy_encoded)
        output_data = scrambler(decoded_scrambled)  # descrambling

        # 3. verification
        bit_errors_after_decoding = np.sum(input_data != output_data)

        status = "PERFECT" if bit_errors_after_decoding == 0 else "DECODER DIED"

        print(f"{errors_inserted:<15} | {bit_errors_after_decoding:<15} | {status}")

if __name__ == "__main__":
    run_experiment()