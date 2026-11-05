# -*- coding: utf-8 -*-
"""
Simulate channel characteristics

Created on Fri Nov  8 13:03:38 2024

@author: drm1g20
"""

import numpy as np

"""
Noises
"""
def amplitude_noise(signal, stddev):
    rnd_samples = np.random.normal(scale=stddev,
                                   size=(len(signal)))
    signal.real += rnd_samples
    
    if np.any(signal.imag):
        rnd_samples = np.random.normal(scale=stddev,
                                       size=(len(signal)))
        signal.imag += rnd_samples
    
    return signal


def phase_noise(signal, stddev):
    o_arg = np.angle(signal)
    
    rnd_samples = np.random.normal(scale=stddev,
                                   size=(len(signal)))
    
    n_arg = o_arg + rnd_samples
    
    signal.real = np.absolute(signal) * np.cos(n_arg)
    if np.any(signal.imag):
        signal.imag = np.absolute(signal) * np.sin(n_arg)
    
    return signal


"""
Signal analysis
"""
def PAPR(signal):
    mean = np.mean(np.power(np.absolute(signal), 2))
    papr = 10 * np.log10(np.max(np.power(np.absolute(signal), 2)) / 
                    np.mean(np.power(np.absolute(signal), 2)))
    
    return mean, np.std(signal), papr


def print_PAPR(signal):
    papr = PAPR(signal)
    print("Mean power:", papr[0])
    print("Stddev:", papr[1])
    print("PAPR:", papr[2])