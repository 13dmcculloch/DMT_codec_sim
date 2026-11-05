# -*- coding: utf-8 -*-
"""
DMT codec.

Created on Thu Nov  7 12:26:20 2024

@author: drm1g20
"""

import numpy as np

import modulations
import channel
import sys

class DMT_Symbol:
    """
    This also allows anything to be loaded into a DMT symbol allowing us
    to find the amble or CP from arbitrary waveform for synchronisation.
    Using the results from the synch functions here, the parent function
    can then load in the correct data for demodulation.
    """
    def __init__(self, waveform,
                 n_subcarriers=32,
                 pilots=(31,),
                 cp_length=32//4):
        self.waveform = waveform  # False if symbol to be modulated
        self.n_subcarriers = n_subcarriers
        self.pilots = pilots
        self.cp_length = cp_length
        
        self.symbol_length = 2 * n_subcarriers + cp_length
        
        self.bv_f = 0  # file object for input data bit vector as INTEGERS
        
        self.bit_loading = 0
        
        self.selmap = False
        self.two_channel_ft = False
        
        self.buffer = 0  # set to work on existing data
        self.amble_bv = 0  # set to make amble symbol
        self.ifft_n = 0
        self.ifft_i = 0  # would be used if doing a two-channel FT
        self.ifft_out = 0
        self.symbol = 0
        
        self.fft_in = 0
        self.fft_out = 0
        self.bv_out = 0
        
        
    """
    Modulation functions
    """
    def _read_in(self):
        if not self.bv_f:
            print("Need to specify bit vector file.")
            return
        
        self.bit_load_size = sum(self.bit_loading.values())
        
        # load appropriate number of carriers for bit loading configuration
        n_read = self.bit_load_size * 2   # includes '\n'
        strbuf = self.bv_f.read(n_read)
        if strbuf == '':
            return
        buffer = np.fromstring(strbuf,
                               sep='\n',
                               dtype=int)
        
        # pad with zeroes if buffer incomplete
        if len(buffer) != n_read and len(strbuf) > 0:
            fullbuf = np.zeros(n_read,
                               dtype=int)
            fullbuf[:len(buffer)] = buffer
            buffer = fullbuf
        
        self.buffer = buffer
        
        return buffer
    
        
    def _bit_load(self):
        """
        Return array of OFDM-buffer modulated data from bv_f according to:
        loading = {
            1: (bpsk carriers):
            2: (qpsk carriers):
            ...
            n: (mod-order loaded carriers)  # see modulations.modulation_order
        }
        """
        if np.any(self.amble_bv):
            buffer = self.amble_bv
        # if we manually set buffer, we won't read from file again
        elif not np.any(self.buffer):
            buffer = self._read_in()
        else:
            buffer = self.buffer
        
        if buffer is None:
            return 'EOF'
            
        # load each carrier with each modulated bits
        ifft_n = np.zeros(self.n_subcarriers,dtype=complex)
        ifft_i = np.zeros(self.n_subcarriers,dtype=complex)
        buffer_pos = 0
        for carrier, order in self.bit_loading.items():
            if carrier == 0:
                print("Invalid carrier at DC.")
                return
            
            """
            Schmidl says:
                The two
                halves of the training symbol are made identical (in time
                order) by transmitting a pseudonoise (PN) sequence on the
                even frequencies, while zeros are used on the odd frequen-
                cies
            """
            if np.any(self.amble_bv) and (carrier % 2) != 0:
                continue
            
            if carrier > self.n_subcarriers - 1:
                print("Invalid subcarrier either at pilot position or \
                      outside.")
                return
            
            ifft_n[carrier] = modulations.modulate_bits(buffer[buffer_pos:buffer_pos+order],
                                                  order,
                                                  invert=self.selmap)

            buffer_pos += order
            
        self.ifft_n = ifft_n
        self.ifft_i = ifft_i
        
        return ifft_n, ifft_i
    
    
    def _pilot(self):
        """
        Inside of constellation for pilot, which takes format of last SC
        1s = normal, 0s = inverted
        """
        # just makes final sc in ifft_h equal to selmap value
        if self.ifft_n[-1] or self.ifft_i[-1]:
            print("Loaded data register incorrectly.")
            return
        
        modulation_order = list(self.bit_loading.values())[-1]
        bits = tuple([1 for b in range(modulation_order)])
        bits_inv = tuple([int(not i) for i in bits])
        if not self.selmap:
            self.ifft_n[-1] = modulations.modulate_bits(bits, 
                                                        modulation_order)
        else:
            self.ifft_n[-1] = modulations.modulate_bits(bits_inv, 
                                                        modulation_order)
        self.ifft_i[-1] = modulations.modulate_bits(bits_inv, 
                                                    modulation_order)

        
    def _hermitian_ifft(self):
        if not np.any(self.ifft_n):
            print("Need to load data register first.")
            return
    
        nh = np.zeros(self.n_subcarriers,
                         dtype=complex)

        n = self.ifft_n
        nh[1:] = np.flip(np.conjugate(n[1:]))

        ifft_bins = np.concatenate((n, nh))

        if np.any(self.amble_bv):
            ifft_bins *= 1.141  # says this in literature for "constant energy"

        # perform IFFT
        self.ifft_out = np.fft.ifft(ifft_bins)

        return self.ifft_out
    
    
    def _add_cp(self):
        symbol = np.zeros(self.n_subcarriers * 2 + self.cp_length,
                          dtype=complex)
        symbol[:self.n_subcarriers*2] = self.ifft_out
        symbol[self.n_subcarriers*2:] = self.ifft_out[:self.cp_length]
        
        self.symbol = symbol
        
        return symbol
        
    
    def _modulate(self):
        # modulate new data from file, or reuse buffer, or exit if file empty
        bl_res = self._bit_load()
        if type(bl_res) == str and bl_res == 'EOF':
            return 'EOF'
        
        # pilot is an odd subcarrier. demodulator will need to do more work
        if not np.any(self.amble_bv):
            self._pilot()
            
        # conjugate, flip, append
        self._hermitian_ifft()
        
        return self._add_cp()
    
    
    def modulate(self):
        """
        Sets/returns self.waveform to the lowest PAPR signal out of norm/inv
        """
        def compare_papr():
            print("Normal PAPR:", papr_normal,
                  "Inverted PAPR:", papr_invert)

        if self.two_channel_ft:
            """
            normal = ifft_out.real
            invert = ifft_out.imag
            """
        else:
            self.selmap=False
            normal = self._modulate()
            if isinstance(normal, str): 
                if normal == 'EOF':
                    return 'EOF'
    
            self.selmap=True
            invert = self._modulate()
        
        papr_normal = channel.PAPR(normal)[2]
        papr_invert = channel.PAPR(invert)[2]
        #compare_papr()
        
        if papr_normal > papr_invert:
            self.waveform = invert
        else:
            self.waveform = normal
            
        return self.waveform
        
        
    """
    Demodulation functions
    """
    def demodulate(self):
        """
        This is run once it is determined that the waveform is a DMT symbol
        """
        if not np.any(self.bit_loading):
            print("Bit loading configuration not loaded.")
            return
        
        self._remove_cp()
        self._hermitian_fft()
        self._recover_selmap()  # from pilot
        bv = self._demodulate()  # according to known bit loading (not adaptive)
        
        return bv
        
        
    def _remove_cp(self):
        self.fft_in = self.waveform[:self.symbol_length - self.cp_length]
        
        return self.fft_in
    
    
    def _hermitian_fft(self):
        if not np.any(self.fft_in):
            print("FFT input register not loaded.")
            return
        
        fft_out = np.fft.fft(self.fft_in)
        
        self.fft_out = fft_out[:len(fft_out)//2]
        
        return fft_out
    
    
    def _recover_selmap(self):
        if not np.any(self.fft_out):
            print("FFT output register not loaded.")
            return
        
        selmap_sym = self.fft_out[-1]
        
        modulation_order = list(self.bit_loading.values())[-1]
        bits = modulations.demodulate_bits(selmap_sym, modulation_order)
        
        self.selmap = not all(bits)  # 1s = normal -> invert = False
    
    
    def _demodulate(self):
        if not np.any(self.fft_out):
            print("FFT output register not loaded.")
            return
        
        bv = []
        sc = 1
        for s in self.fft_out[1:len(self.fft_out)-1]:
            bits = modulations.demodulate_bits(s, self.bit_loading[sc],
                                               invert=self.selmap)
            bv.append(bits)
            sc += 1
        
        self.bv_out = bv
        self.symbols_out = self.fft_out[1:len(self.fft_out)-1]
        
        return bv
    
    
    """
    Correlation functions
    """
    def _is_correlated(self, correlation, method, threshold):
        if method == 'threshold':
            # is any part of the correlation function over a threhsold?
            correlation_threshold = threshold  # have to do this prudently
            
            result = np.where(correlation > correlation_threshold)[0]
            
            return result
            
        elif method == 'peak':
            # is the peak-to-average of the correlation over a threshold?
            peak_threshold = threshold
            peak = np.max(correlation) / np.mean(correlation)
            
            if peak > peak_threshold:
                return True
        
        else:
            print("Unknown correlation method.")
            return False
        
        return False
        
    
    def is_amble(self, method='threshold', threshold=0.3):
        l = len(self.waveform) // 2 - self.cp_length // 2
        waveform_low = self.waveform[:l]
        waveform_high = self.waveform[l:]
        
        correlation = np.correlate(waveform_low, waveform_high,'full')
        
        res = self._is_correlated(correlation, method, threshold)
        
        return res
    
    
    def is_cp(self, method='threshold', threshold=0.3):
        # return location of CP if we know it is present
        cp = self.waveform[self.cp_length:]
        
        correlation = np.correlate(cp, 
                                   self.waveform, 
                                   'full')
        
        correlation = np.correlate(self.waveform, 
                                   self.waveform[self.cp_length:len(self.waveform)-self.cp_length],                       
                                   'full')
        
        return self._is_correlated(correlation, method, threshold)
        

def generate_amble_bv(filename,
                      modulation_format='bpsk',
                      n_subcarriers=32,
                      cp_length=32//4):
    modulation_order = modulations.modulation_order(modulation_format)
    bv_len = modulation_order * n_subcarriers - 2 # DC and pilot
    bv = np.random.randint(2,size=bv_len)
    np.savetxt(filename, np.transpose(bv))
    
    return bv
    

def generate_flat_bit_loading(modulation_format,
                              n_subcarriers=32):
    r = range(1,n_subcarriers-1)
    modulation_order = modulations.modulation_order(modulation_format)
    if modulation_order is None:
        print("Failed to generate flat bit loading.")
        return

    bit_loading = {c: modulation_order for c in r}
    
    return bit_loading


def amble(amble_bv,
          bit_loading,
          n_subcarriers=32,
          cp_length=32//4):
    amble_symbol = DMT_Symbol(False,
                              n_subcarriers=n_subcarriers,
                              cp_length=cp_length)
    amble_symbol.bit_loading = bit_loading
    amble_symbol.amble_bv = amble_bv
    
    amble_symbol.modulate()
    
    amble_point = amble_symbol.is_amble(method='threshold', threshold=0.1)
    if np.any(amble_point):
        return amble_symbol
    else:
        print("Could not determine if amble is correctly generated.")
    
    return amble_symbol


def check_amble_length(amble, n_subcarriers, cp_length):
    amble_len = len(amble.waveform)
    if amble_len != n_subcarriers*2 + cp_length:
        print("Invalid amble length:", amble_len)
        return False
    return True


def DMT_modulate(bv_f,
                 amble,
                 bit_loading,
                 n_subcarriers=32,
                 pilots=(31,),
                 cp_length=32//4):
    if amble:
        check_amble_length(amble, n_subcarriers, cp_length)
    
    if not bv_f.readable():
        print("Bit vector input file is not readable.")
        return
    
    signal = []
    amble_remaining = 0
    n_symbol = 0
    while True:
        if amble and amble_remaining < 1:
            signal.append(amble.waveform.tolist())
            amble_remaining = 100  # used in Lee 2008
        
        sym = DMT_Symbol(False,
                         n_subcarriers=n_subcarriers,
                         pilots=pilots,
                         cp_length=cp_length)
        sym.bv_f = bv_f
        sym.bit_loading = bit_loading
        mod_res = sym.modulate()
        
        if isinstance(mod_res, str):
            if mod_res == 'EOF':
                break
            
        signal.append(mod_res.tolist())
        
        amble_remaining -= 1
        n_symbol += 1
        
        
    return np.ravel(np.array(signal))


def load_floats(f, n_lines):
    floats = []
    n_read = 0
    for n in range(n_lines):
        line = f.readline()
        if line == '':
            break

        floats.append(complex(line).real)
        
        n_read += 1

    return np.array(floats), n_read


def seek_lines(f, n):
    # seek file to nth line
    for n in range(n):
        f.readline()


def DMT_demodulate(Rx_f,
                   amble,
                   amble_bv,
                   bit_loading,
                   n_subcarriers=32,
                   pilots=(31,),
                   cp_length=32//4,
                   return_symbols=False):
    
    if amble:
        check_amble_length(amble, n_subcarriers, cp_length)
    
    if not Rx_f.readable():
        print("Received signal input file is not readable.")
        return
    
    rx_bits = []
    rx_syms = []
    
    """
    Demodulating with synch
    """
    symbol_length = n_subcarriers * 2 + cp_length
    
    eof = False
    """
    Implemenatation of synch:
        From the start, walk through until the amble is found, then
        set the offset to this value and start demodulating normally.
        CP timing TBC
    """
    # Unsynched scenario
    at_amble = not amble
    repeat_bin = 39
    pos_f = 0
    symbol = DMT_Symbol(0,
                        n_subcarriers=n_subcarriers,
                        pilots=pilots,
                        cp_length=cp_length)
    symbol.bit_loading = bit_loading
    
    def test_input(n_read):  # Think it needs passing b/c of assignment
        if n_read != symbol_length:
            #print("Reached EOF, but read", n_read, "lines.")
            return 'eof'
            
        if n_read == 0:
            return False
        
        while n_read > 0 and n_read != symbol_length:
            np.concatenate((symbol_signal,np.zeros(1,dtype=float)))
            n_read -= 1
        
        return True
    
    # Synch using amble. We don't care about the demodulated amble value yet.
    # If amble passed to function, deal with the amble, otherwise skip
    while not at_amble and not eof and amble:
        # Load in the symbol
        Rx_f.seek(0)
        seek_lines(Rx_f, pos_f)  # Wind her back for next test waveform
        
        symbol_signal, n_read = load_floats(Rx_f, symbol_length)
        ti = test_input(n_read)
        if ti == 'eof' or not ti:
            break
        else:
            symbol.waveform = symbol_signal
        
        # Is it an amble?
        correl = symbol.is_amble(method='threshold',
                                   threshold=0.3)
        
        """
        f_pos = pos_f
        if f_pos > 190 and f_pos < 280:
            print(str(f_pos) + ':' + str(correl))
        """
        # demodulate ambles detected near correlation peak
        if repeat_bin in correl:
            demod_res = symbol.demodulate()
            if isinstance(demod_res, str):
                break

            possible_amble = np.ravel(demod_res[1::2])  # amble in bv file
            amble_bv_short = amble_bv[:-len(amble_bv)//2-1]
            # matching amble value confirms amble detected
            if np.all(possible_amble == amble_bv_short):
                at_amble = True
    
        pos_f += 1
        
    if at_amble:
        print("Found the amble at symbol", pos_f)
    else:
        print("Did not find an amble.")
        return

    
    # Synched demodulation using just amble synch
    while not eof:
        # Load in waveform with normal procedure
        symbol_signal, n_read = load_floats(Rx_f, symbol_length)
        ti = test_input(n_read)
        if ti == 'eof' or not ti:
            break
        else:
            symbol.waveform = symbol_signal
        
        correl = symbol.is_amble(method='threshold',
                                   threshold=0.1)
        
        if len(correl) == 1:
            if correl[0] == 32:
                at_amble = True
    
        
        demod_res = symbol.demodulate()
        if isinstance(demod_res, str):
            break
    
        # only bother comparing amble if at the amble
        if at_amble:
            possible_amble = np.ravel(demod_res[1::2])  # amble in bv file
            amble_bv_short = amble_bv[:-len(amble_bv)//2-1]
            if np.all(possible_amble == amble_bv_short):
                # omit this symbol from the data output
                continue
        
        if return_symbols:
            rx_syms.append(symbol.symbols_out)
        rx_bits.append(demod_res)
        
    if return_symbols:
        return np.ravel(np.array(rx_syms))
    return np.ravel(np.array(rx_bits))

"""
Procedural versions of the functions are in the previous versions of the DMT
code.
"""
