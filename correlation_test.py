# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:24:25 2024

@author: drm1g20
"""

import numpy as np
import matplotlib.pyplot as plt#
import sys

import DMT_codec


compare_amble = False  # test amble correlator
compare_cp = True  # test cp correlator

amble_filename = 'amble'

amble_bv = np.genfromtxt(amble_filename,
                         delimiter='\n',
                         dtype=int)

amble = DMT_codec.DMT_Symbol(DMT_codec.amble(amble_bv))
    
sym_wav = np.random.randint(30,size=len(amble.waveform)) / 100
sym_wav -= max(sym_wav) / 2
sym = DMT_codec.DMT_Symbol(sym_wav)

if compare_amble:
    fig,ax = plt.subplots()
    ax.plot(amble.waveform,label='amble')
    #ax.plot(sym.waveform,label='noise')
    #ax.plot(amble.is_amble(),label='correlation:amble')
    #ax.plot(sym.is_amble(),label='correlation:noise')
    ax.set_ylim((-0.4,0.4))
    ax.legend()
    plt.show()
    
if compare_cp:
    print(amble.is_amble())
    print(amble.is_cp(threshold=0.5))
    print(sym.is_amble())
    print(sym.is_cp(threshold=0.5))
    sys.exit(0)
    
    fig,ax = plt.subplots()
    ax.plot(amble.waveform,label='amble')
    #ax.plot(sym.waveform,label='noise')
    ax.plot(amble.is_cp(),label='correlation:amble')
    #ax.plot(sym.is_cp(),label='correlation:noise')
    ax.set_ylim((-0.4,0.4))
    #ax.set_xlim((60,80))
    ax.legend()
    plt.show()
