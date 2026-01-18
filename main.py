from encoder import ConvolutionalEncoder
from decoder import Decoder

if __name__ == "__main__":
    e = ConvolutionalEncoder(128)
    d = Decoder(e.encode)