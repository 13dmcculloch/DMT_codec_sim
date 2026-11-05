# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:34:32 2024

@author: drm1g20
"""

import OFDM_codec2
import prbs
import modulations

import sys
import numpy as np
import matplotlib.pyplot as plt 

waveform_length = int(32e6)  # keep short while coding
sps = int(10e9)

waveform_duration = waveform_length / sps

carrier_freq = int(3000e6)
upsampling_factor = sps // carrier_freq
upsampling_factor = 14

prbs_bitarray = prbs.PRBS((15,14))
prbs_symbols = modulations.modulate_data(prbs_bitarray,
                                         modulation_format='bpsk')

Tx_signal, cp_length = OFDM_codec2.OFDM_modulate(prbs_symbols)

signal_length = len(Tx_signal)

# Upsampling
original_s = np.arange(signal_length)
new_s = np.arange(signal_length * upsampling_factor) / upsampling_factor

upsampled_signal = np.interp(new_s, original_s, Tx_signal)
upsampled_signal_length = len(upsampled_signal)
upsampled_signal_duration = upsampled_signal_length / sps

# Quadrature mixing
time_bins = np.linspace(0,upsampled_signal_duration,upsampled_signal_length)
sinusoid = np.sin(time_bins * 2 * np.pi * carrier_freq)
cosinusoid = np.cos(time_bins * 2 * np.pi * carrier_freq)

mixed_signal = np.real(upsampled_signal) * sinusoid + \
                np.imag(upsampled_signal) * cosinusoid
                
print(upsampled_signal_length)


fig,ax = plt.subplots()

ax.plot(time_bins * 1e6, np.abs(upsampled_signal), label='Original signal')
ax.plot(time_bins * 1e6, mixed_signal, label='Mixed signal')

ax.set_xlabel("Time (us)")
ax.set_ylabel("something")

ax.set_xlim(0,0.01)

ax.legend()

plt.show()

