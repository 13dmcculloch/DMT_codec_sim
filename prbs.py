# -*- coding: utf-8 -*-
"""
PRBS generator from two taps
Created on Thu Oct 24 12:21:47 2024

@author: drm1g20
"""

import numpy as np
import os

def PRBS(polynomials: (int, int)):
    start = 1
    state = start
    bitstr = [start]
    length = 2**(max(polynomials)) - 1
    for i in range(length):
        newbit = (((state >> polynomials[0] - 1) ^ (state >> polynomials[1] - 1)) & 1)
        bitstr.append(newbit)
        state = ((state << 1) | newbit) & length
    bitarray = np.array(bitstr)
    return bitarray


def generate_save(polynomials):
    # use it in the REPL to make arbitrary PRBS's with two coefficients
    for p in polynomials:
        filename = 'PRBS' + str(p[0])
        np.savetxt(filename, np.transpose(PRBS(p).astype(int)),
                   fmt='%d')
    

def load_PRBS(filename):
    return np.genfromtxt(filename)


def filesize(filename):
    f_length = os.stat(filename).st_size
    print(filename + " file size: " + str(int(f_length) // 1024) + 'K')
    