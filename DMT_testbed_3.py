# -*- coding: utf-8 -*-
"""
Creates unsynched signal and demodulates it.

Todo:
    Do something about the final symbol being of such high PAPR
    Can we have a closing amble then follow it with better padded data?
    Closing amble says: "the next 5 subcarriers are data, but the remainder
    are all noise, so the EOF is straight after the 5th subcarrier"
    
Created on Mon Nov 18 17:04:48 2024

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
# DMT modulation
n_subcarriers = 32
cp_length = n_subcarriers // 4
modulation_format = 'qpsk'

# input data
bits_filename = 'PRBS7'
Tx_filename = 'DMT_Tx'  # signal transmitted
Rx_filename = 'DMT_Rx'  # signal from DSO simulation
amble_32_filename = 'amble_5PAPR22_32sc_qpsk'  # bit vector of amble

# time noises
noise_length_pre = 200  # number of samples of noise before transmission
noise_length_post = 200
noise_amplitude = 0.25

# additive noises

# simulation parameters for testing different parts of process
apply_amble = True
apply_time_noise = True
apply_noise = True
EVM = True

"""
Modulation format - default bit loading
"""
bit_loading = DMT_codec.generate_flat_bit_loading(modulation_format,
                                                  n_subcarriers=n_subcarriers)

"""
Amble generation
"""
if n_subcarriers == 32 and modulation_format == 'qpsk':
    amble_filename = amble_32_filename  # use amble of known PAPR for config
else:
    # create new amble for this number of subcarriers
    amble_filename = 'amble'

if n_subcarriers == 32 and modulation_format == 'qpsk':
    amble_bv = np.fromstring(open(amble_filename,'r').read(), 
                             dtype=float,  # wanted ints in this file...
                             sep='\n').astype(int)
else:
    amble_bv = DMT_codec.generate_amble_bv(amble_filename,
                                          modulation_format=modulation_format,
                                          n_subcarriers=n_subcarriers,
                                          cp_length=cp_length)

if not apply_amble:
    amble = False
else:
    amble = DMT_codec.amble(amble_bv, bit_loading,
                                   n_subcarriers=n_subcarriers,
                                   cp_length=cp_length)

"""
Signal generation
"""
bv_f = open(bits_filename, 'r')
Tx_bv = np.fromstring(bv_f.read(),dtype=int,sep='\n')
bv_f.seek(0)

Tx = DMT_codec.DMT_modulate(bv_f, amble, bit_loading,
                            n_subcarriers=n_subcarriers,
                            cp_length=cp_length)

bv_f.close()

"""
Tx signal analysis
"""
# print bit vector if it is short enough to display
if bits_filename == 'PRBS7':
    print("Tx data:\n", Tx_bv)
    
# check for fourier messing up
if np.any(Tx.imag):
    print("Tx has imaginary part. Something might be wrong.")

np.savetxt(Tx_filename, np.transpose(Tx).real,delimiter='\n')

print("\nTx channel characteristic:")
channel.print_PAPR(Tx)

"""
Channel - simulating a DSO
"""
# additive noises here
if apply_noise:
    Tx = channel.amplitude_noise(Tx, 0.01)
    
# generate pre/appended "time" noise and remove DC
# this noise simulates the start and end of capture from a scope
if apply_time_noise:
    noise_signal_pre = np.random.random(noise_length_pre) * noise_amplitude * 2
    noise_signal_post = np.random.random(noise_length_post) * noise_amplitude * 2
    noise_signal_pre -= np.mean(noise_signal_pre)
    noise_signal_post -= np.mean(noise_signal_post)
    
    # prepend and append time noises
    Rx = np.concatenate((noise_signal_pre,
                        Tx,
                        noise_signal_post))
    
np.savetxt(Rx_filename, np.transpose(Rx).real,delimiter='\n')

"""
Rx signal analysis
"""
print("\nRx channel characteristic:")
channel.print_PAPR(Rx)

"""
Demodulation
"""
Rx_f = open(Rx_filename, 'r')

Rx_data = Rx_f.readlines()
Rx_signal = [complex(line) for line in Rx_data]

Rx_f.seek(0)

Rx_padded = DMT_codec.DMT_demodulate(Rx_f, 
                                    amble, 
                                    amble_bv,
                                    bit_loading,
                                    n_subcarriers=n_subcarriers,
                                    cp_length=cp_length,
                                    return_symbols=EVM)
Rx_f.close()

if Rx_padded is None:
    print("Demodulation fail.")
    sys.exit(0)

Rx_padded = Rx_padded[:len(Tx_bv)//2]
print(Rx_padded)    
if EVM:
    print(modulations.EVM(Rx_padded, modulation_format))
    fig,ax = plot_generics.plot_setup("After simulated transmission", 'sample','')
    plot_generics.plot_constellation(ax, Rx_padded, label='Symbols')
    ax.set_xlim((-1.5,1.5))
    ax.set_ylim((-1.5,1.5))
    plot_generics.plot_show(ax)
    sys.exit(0)

Rx_bv = Rx_padded[:len(Tx_bv)]

"""
Demodulated analysis
"""
if bits_filename == 'PRBS7':
    print("Rx data:", Rx_bv)
    
if np.any(Rx_bv != Tx_bv):
    print("Demodulated data is, I mean are, different!")
else:
    print("Success! Demodulated data matches input data.")

"""
Plotting
"""
fig,ax = plot_generics.plot_setup("After simulated transmission", 'sample','')
ax.plot(np.real(Rx_signal),label='Rx')
#ax.set_xlim((260,340))
plot_generics.plot_show(ax)
