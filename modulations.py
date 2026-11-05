# -*- coding: utf-8 -*-
"""
Modulation and analysis

Created on Thu Oct 24 12:17:56 2024

@author: drm1g20
"""

import numpy as np

"""
Modulation tables
"""
QPSK = {
        (0, 0): complex(0.707, 0.707),
        (0, 1): complex(0.707, -0.707),
        (1, 1): complex(-0.707, -0.707),
        (1, 0): complex(-0.707, 0.707)
        }

QPSK_i = {
        (1, 1): complex(0.707, 0.707),  # invert it, don't cycle
        (1, 0): complex(0.707, -0.707),
        (0, 0): complex(-0.707, -0.707),
        (0, 1): complex(-0.707, 0.707)
        }

QPSK_i = {
        (1, 0): complex(0.707, 0.707),  # cycle it, don't just invert
        (0, 0): complex(0.707, -0.707),
        (0, 1): complex(-0.707, -0.707),
        (1, 1): complex(-0.707, 0.707)
        }
"""
Bread and butter
"""
def modulate_data(data, modulation_format='bpsk', invert=False):
    """
    Map the int/bool/ndarray data to complex ndarray of symbols
    """
    if modulation_format == 'bpsk' or modulation_format == 1:
        modulated_data = np.array(np.zeros(len(data),dtype=complex))
        if not invert:
            for i, bit in enumerate(data):
                modulated_data[i] = complex(np.power(-1, bit), 0)
        else:
            for i, bit in enumerate(data):
                modulated_data[i] = complex(np.power(-1, 1 - bit), 0)
            
    elif modulation_format == 'qpsk' or modulation_format == 2:
        modulated_data_length = len(data) // 2
        modulated_data = np.array(np.zeros(modulated_data_length),
                                  dtype=complex)
        if not invert:
            for i in range(1,len(data),2):
                modulated_data[i//2] = QPSK[data[i-1], data[i]]
        else:
            for i in range(1,len(data),2):
                modulated_data[i//2] = QPSK_i[data[i-1], data[i]]
        
    else:
        print("Unrecognised modulation format")
        return
        
    return modulated_data


def modulate_bits(bits, modulation_order, invert=False):
    """
    Modulate tuple of bits to complex symbol
    """
    #print(bits)
    if len(bits) != modulation_order:
        print("modulate_bits: Not enough bits for modulation order:", 
              modulation_order)
        return
    
    if modulation_order == 1:
        bit = bits[0]
        if not invert:
            symbol = complex(np.power(-1, bit), 0)
        else:
            symbol = complex(np.power(-1, 1 - bit), 0)
            
    elif modulation_order == 2:
        if not invert:
            symbol = QPSK[bits[0],bits[1]]
        else:
            symbol = QPSK_i[bits[0],bits[1]]
        
    else:
        print("modulate_bits: Unrecognised modulation format")
        return
        
    return symbol


def demodulate_bits(symbol, modulation_order, invert=False):
    demodulated_bits = 0
    
    if modulation_order == 1:
        cur_sym = symbol
        
        if not invert:
            if cur_sym.real > 0:
                demodulated_bits = (0,)
            else:
                demodulated_bits = (1,)
        else:
            if cur_sym.real < 0:
                demodulated_bits = (0,)
            else:
                demodulated_bits = (1,)
            
    elif modulation_order == 2:
        if not invert:
            constellation = QPSK
        else:
            constellation = QPSK_i
            
        demodulated_bits = (0, 0)
        min_dist = np.Inf
        for bits, point in constellation.items():
            dist = distance_from_ref(symbol, point)
            if dist < min_dist:
                demodulated_bits = bits
                min_dist = dist
                
        return demodulated_bits
    
    else:
        print("Unrecognised modulation format")
        return


def demodulate_data(symbols, modulation_format='qpsk'):
    """
    This demodulation function does not make decisions
    """
    conventional = False
    
    if modulation_format == 'bpsk':
        demodulated_data_length = len(symbols)
        demodulated_data = np.array(np.zeros(demodulated_data_length))
        
        for i in range(demodulated_data_length):
            cur_sym = symbols[i]
            
            if cur_sym.real > 0:
                demodulated_data[i] = 0
            else:
                demodulated_data[i] = 1
        
    elif modulation_format == 'qpsk':
        demodulated_data_length = len(symbols) * 2
        demodulated_data = np.array(np.zeros(demodulated_data_length))
        for i in range(0,demodulated_data_length,2):
            if conventional:
                high, low = list(QPSK.keys())\
                            [list(QPSK.values())\
                             .index(symbols[i//2])]
            else:
                cur_sym = symbols[i//2]
                
                if cur_sym.real > 0:
                    sym_real = 0.707
                else:
                    sym_real = -0.707
                    
                if cur_sym.imag > 0:
                    sym_imag = 0.707
                else:
                    sym_imag = -0.707
                    
                high, low = list(QPSK.keys())\
                            [list(QPSK.values())\
                             .index(complex(sym_real, sym_imag))]
            
            demodulated_data[i] = high
            demodulated_data[i + 1] = low
            
    else:
        print("Unrecognised modulation format")
        return
        
    return demodulated_data


"""
I am not smart enough for metaprogramming
"""
def modulation_order(modulation_format):
    """
    Return BPSK, QPSK modulation order from modulation format id
    """
    modulation_order = 0
    if modulation_format == 'bpsk':
        modulation_order = 1
    elif modulation_format == 'qpsk':
        modulation_order = 2
    else:
        print("modulation_order: Unrecognised modulation format.")
        return
    
    return modulation_order


"""
BER analysis
"""
def _count_errors(Tx_data, Rx_data):
    if len(Tx_data) != len(Rx_data):
        print("_count_errors: Bit vectors are of unequal length.")
        return
    
    return len(Rx_data) - np.sum(Rx_data == Tx_data)


def BER(Tx_data, Rx_data):
    return _count_errors(Tx_data, Rx_data) / len(Tx_data)


"""
EVM analysis
"""
def distance_from_ref(v, ref):
    e_real = ref.real - v.real
    e_imag = ref.imag - v.imag
    
    distance = abs(complex(e_real, e_imag))
    
    return distance


def EVM(Rx_symbols, modulation_format):
    total_evm = 0
    if modulation_format == 'bpsk':
        for s in Rx_symbols:
            # snap to correct symbol
            if s.real > 0:
                ref = complex(1,0)
            else:
                ref = complex(-1,0)
            
            # errors of components
            total_evm += distance_from_ref(s, ref)  # magnitude of symvec is 1
            
    elif modulation_format == 'qpsk':
        for s in Rx_symbols:
            dist = [distance_from_ref(s, ref) for ref in QPSK.values()]
            total_evm += min(dist)
            
    else:
        print("Unrecognised modulation format.")
        
    return 100 * total_evm / len(Rx_symbols)