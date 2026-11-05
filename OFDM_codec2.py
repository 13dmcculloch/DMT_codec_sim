import modulations
import prbs

import numpy as np

def OFDM_modulate(symbols,
                    n_subcarriers=32,
                    pilots=()
                    ):
    """
    Modulate symbols onto n_subcarriers subcarriers
    """
    # Arithmetic
    cp_length = n_subcarriers//4
    n_pilots = len(pilots)
    n_input_symbols = len(symbols)
    ofdm_symbol_length = n_subcarriers + cp_length 
    n_data_subcarriers = n_subcarriers - n_pilots 
    n_ofdm_symbols = int(np.ceil(n_input_symbols / n_data_subcarriers))
    
    output_signals = np.array(np.zeros(n_ofdm_symbols * ofdm_symbol_length),
                            dtype=complex)

    for i in range(n_ofdm_symbols):
        ofdm_symbol = np.array(np.zeros(ofdm_symbol_length),
                              dtype=complex)
        blk = np.array(np.zeros(n_subcarriers),
                        dtype=complex)

        # S/P ignore pilots for now
        blk = symbols[i * n_data_subcarriers:(i + 1) * n_data_subcarriers]

        blk = np.fft.ifft(blk)

        # P/S and add CP
        ofdm_symbol[:len(blk)] = blk
        ofdm_symbol[len(blk):len(blk) + cp_length] = blk[:cp_length]

        output_signals[i * ofdm_symbol_length:(i + 1) * ofdm_symbol_length] = ofdm_symbol

    return output_signals, cp_length


def OFDM_demodulate(Rx,
                    n_subcarriers=32,
                    pilots=(),
                    cp_length=8):
    """
    Demodulate OFDM signal into array of symbols
    """
    rx_length = len(Rx)
    n_pilots = len(pilots)
    n_data_subcarriers = n_subcarriers - n_pilots
    ofdm_symbol_length = n_subcarriers + cp_length
    n_ofdm_symbols = int(np.ceil(rx_length / ofdm_symbol_length))
    output_length = n_ofdm_symbols * n_data_subcarriers

    print("Demodulate:")
    print("rx_length:",rx_length)
    print("n_ofdm_symbols:",n_ofdm_symbols)

    output = np.array(np.zeros(output_length),
                        dtype=complex)

    for i in range(n_ofdm_symbols):
        blk = np.array(np.zeros(n_subcarriers),
                        dtype=complex)

        # S/P and CP removal
        blk = Rx[i * (n_subcarriers + cp_length):(i+1) * n_subcarriers + i * cp_length]

        blk = np.fft.fft(blk)

        output[i * n_data_subcarriers:(i + 1) * n_data_subcarriers] = blk

    return output
