import numpy as np

class FECsim:
    def __init__(self):
        # 802.11ah params: K=7, R=1/2
        self.poly_a = 0o133  # 1011011
        self.poly_b = 0o171  # 1111001

        # precalc trellis stuff, basically a map for the decoder to move around
        self.trellis_next_state = np.zeros((64, 2), dtype=int)
        self.trellis_output = np.zeros((64, 2), dtype=int)
        self._build_trellis()

    def _build_trellis(self):
        # builds state transition map
        for state in range(64):
            for bit in [0, 1]:
                next_state = ((state << 1) | bit) & 0x3F
                full_reg = (state << 1) | bit
                out_a = bin(full_reg & self.poly_a).count('1') % 2
                out_b = bin(full_reg & self.poly_b).count('1') % 2
                self.trellis_next_state[state][bit] = next_state
                self.trellis_output[state][bit] = (out_a << 1) | out_b

    def scrambler(self, bits, seed=0b1011101):
        # scrambler with polynomial x^7 + x^4 + 1
        state = seed
        output = np.zeros_like(bits)
        for i in range(len(bits)):
            feedback = ((state >> 6) ^ (state >> 3)) & 1
            output[i] = bits[i] ^ feedback
            state = ((state << 1) | feedback) & 0x7F
        return output

    def encoder(self, bits):
        # convolutional encoder
        state = 0
        encoded = []
        # adding 6 zeros at the end (tail bits) to close coding properly
        padded_bits = np.concatenate((bits, [0] * 6))

        for bit in padded_bits:
            full_reg = (state << 1) | bit
            out_a = bin(full_reg & self.poly_a).count('1') % 2
            out_b = bin(full_reg & self.poly_b).count('1') % 2
            encoded.extend([out_a, out_b])
            state = full_reg & 0x3F
        return np.array(encoded, dtype=int)

    def decoder(self, encoded_bits):
        """
        viterbi decoder
        logic: looks for a path in trellis that differs the least
        from what we received (minimizes errors)
        """
        n_steps = len(encoded_bits) // 2
        path_metrics = np.full(64, 1e6)  # path costs (big number = infinity)
        path_metrics[0] = 0  # starting at state 0
        traceback = np.zeros((n_steps, 64), dtype=int)  # path memory

        # forward step (calculating costs)
        for t in range(n_steps):
            rx_pair = (encoded_bits[2 * t] << 1) | encoded_bits[2 * t + 1]
            new_metrics = np.full(64, 1e6)

            for state in range(64):
                if path_metrics[state] >= 1e6: continue

                for bit in [0, 1]:
                    next_s = self.trellis_next_state[state][bit]
                    expected_out = self.trellis_output[state][bit]

                    # how many bits are different? (hamming distance)
                    # this is the heart of error fixing
                    cost = bin(rx_pair ^ expected_out).count('1')

                    if path_metrics[state] + cost < new_metrics[next_s]:
                        new_metrics[next_s] = path_metrics[state] + cost
                        traceback[t][next_s] = state
            path_metrics = new_metrics

        # backward step (recovering data)
        decoded = []
        state = 0  # we know we end at state 0 cause of tail bits
        for t in range(n_steps - 1, -1, -1):
            prev = traceback[t][state]
            # checking if we came from bit 0 or 1
            bit = 0 if (((prev << 1) | 0) & 0x3F) == state else 1
            decoded.append(bit)
            state = prev

        return np.array(decoded[::-1][:-6])  # reverse and cut off tail bits

    def inject_errors(self, bits, error_count):
        # injects given amount of errors in random places
        corrupted = bits.copy()
        indices = np.random.choice(len(bits), error_count, replace=False)
        for idx in indices:
            corrupted[idx] = 1 - corrupted[idx]  # bit flip (0->1 or 1->0)
        return corrupted