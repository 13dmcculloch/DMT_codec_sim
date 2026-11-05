# -*- coding: utf-8 -*-
"""
Classful DMT testbed

To do:
    De/modulates a synched waveform.

Created on Mon Nov 11 12:21:21 2024

@author: drm1g20
"""

import numpy as np
import matplotlib.pyplot as plt

import DMT_codec
import sys
import plot_generics
import channel
import modulations
import prbs


"""
Parameters
"""
bits_filename = 'PRBS7'
amble_filename = 'amble'
amble_32 = 'amble_5PAPR22_32sc_qpsk'
Tx_filename = 'DMT_signal'

modulation_format = 'qpsk'

# Set to False for BER.
EVM = False

n_subcarriers = 32
cp_length = n_subcarriers // 4


"""
Signal, amble generation
"""
bit_loading = DMT_codec.generate_flat_bit_loading(modulation_format,
                                                  n_subcarriers=n_subcarriers)


amble_bv = DMT_codec.generate_amble_bv(amble_filename,
                                      modulation_format=modulation_format,
                                      n_subcarriers=n_subcarriers,
                                      cp_length=cp_length)

if n_subcarriers == 32:
    amble_bv = np.fromstring(open(amble_32,'r').read(), dtype=float, sep='\n')
else:
    amble_bv = np.fromstring(open(amble_filename,'r').read(), dtype=float, sep='\n')

amble = DMT_codec.amble(amble_bv, bit_loading,
                               n_subcarriers=n_subcarriers,
                               cp_length=cp_length)

bv_f = open(bits_filename, 'r')
Tx_bv = np.fromstring(bv_f.read(),dtype=int,sep='\n')
bv_f.seek(0)

Tx = DMT_codec.DMT_modulate(bv_f, False, bit_loading,
                            n_subcarriers=n_subcarriers,
                            cp_length=cp_length)

bv_f.close()

"""
Signal analysis and whatever
"""
# print bit vector if it is short enough to display
if bits_filename == 'PRBS7':
    print("Tx data:\n", Tx_bv)
    
print("\nTx channel characteristic:")
channel.print_PAPR(Tx)
if np.any(Tx.imag):
    print("Tx has imaginary part. Something might be wrong.")
np.savetxt(Tx_filename, np.transpose(Tx).real,delimiter='\n')


"""
Demodulation
"""
Rx_f = open(Tx_filename, 'r')
"""
Rx_data = Rx_f.readlines()
Rx_signal = [complex(line) for line in Rx_data]
"""
#Rx_f.seek(0)

Rx = DMT_codec.DMT_demodulate(Rx_f, amble, bit_loading,
                              n_subcarriers=n_subcarriers,
                              cp_length=cp_length)
Rx_f.close()

if bits_filename == 'PRBS7':
    print("Rx data:", Rx)

Rx_bv = Rx[:len(Tx_bv)]  # trim padded zeroes

"""
Evaluation
"""
if np.any(Rx_bv != Tx_bv):
    print("Demodulated data is different!")
else:
    print("Success! Demodulated data matches input data.")

"""
Plotting
"""
fig,ax = plot_generics.plot_setup("", 'sample','')
ax.plot(np.absolute(Tx),label='Tx')
plot_generics.plot_show(ax)
