import os 
import numpy as np 
import argparse 
import math
from scipy.signal import chirp
from scipy.ndimage import uniform_filter1d 
from scipy.fft import fft, fftfreq, fftshift
import time
from tqdm import tqdm
import random
import sys
import pickle

def create_parser(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--sf', type=int, help='The spreading factor.') 
    parser.add_argument('--bw', type=int, default=125000, help='The bandwidth.') 
    parser.add_argument('--fs', type=int, default=1000000, help='The sampling rate.') 
    parser.add_argument('--data_dir', type=str, help='Choose the root path to rf signals.') 
    parser.add_argument('--snr', type=float, default=-15, help='Choose the SNR of rf signals.') 
    parser.add_argument('--rep', type=int, default=1, help='repitition on each symbol') 
    parser.add_argument('--debug_upsampling', type=int, default=100, help='upsampling factor of FFT process') 
    parser.add_argument('--debug_search_step', type=int, default=4, help='search steps for phase alignment of the two FFT peaks') 
    return parser 

if __name__ == '__main__':  
    parser = create_parser()
    opts = parser.parse_args()
    assert os.path.exists(opts.data_dir), 'data directory not found'
    opts.n_classes = 2 ** opts.sf
    nsamp = opts.fs * opts.n_classes // opts.bw
    t = np.linspace(0, nsamp / opts.fs, nsamp) 
    chirpI1 = chirp(t, f0=opts.bw/2, f1=-opts.bw/2, t1=2** opts.sf / opts.bw , method='linear', phi=90) 
    chirpQ1 = chirp(t, f0=opts.bw/2, f1=-opts.bw/2, t1=2** opts.sf / opts.bw, method='linear', phi=0) 
    downchirp = chirpI1+1j*chirpQ1 

    ACC = np.zeros((opts.n_classes,), dtype=int) 
    SUM = np.zeros((opts.n_classes,), dtype=int)
    t1 = time.time()
    t0 = time.time()
    tstep = 0
    files = []

    pathfiles = [(filename, os.path.join(root, filename)) for root, dirs, files in os.walk(opts.data_dir) for filename in files if filename[-4:] == '.mat' and os.path.basename(os.path.normpath(root)) != 'raw']
    pbar = tqdm(pathfiles)
    for filename, filepath in pbar:
            symbol_idx = int(filename.split('_')[1]) % opts.n_classes

            fid = open(filepath, 'rb') 
            chirp_raw = np.fromfile(fid, np.float32, nsamp * 2) 
            chirp_raw = chirp_raw[::2] + chirp_raw[1::2]*1j 

            A = [np.mean(abs(chirp_raw)[max(0, i - nsamp//4):min(nsamp, i+nsamp//4+1)]) for i in range(nsamp)]
            chirp_temp = chirp_raw[A >= max(A)/2] 
            amp_sig = np.mean(np.abs(chirp_temp))
            chirp_raw /= amp_sig
             
            #add noise
            for rep in range(opts.rep):
                amp = math.pow(0.1, opts.snr/20) 
                noise = amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp)
                chirp_data = chirp_raw + noise
                chirp_data = chirp_data * downchirp
                fft_raw = fft(chirp_data, len(chirp_data)*opts.debug_upsampling)
                target_nfft=opts.n_classes*opts.debug_upsampling
                signal=fft_raw
                cut1 = signal[:target_nfft]
                cut2 = signal[-target_nfft:]
                cut1=np.array(cut1)
                cut2=np.array(cut2)
        
                # search for phase difference to add up the peaks
                comp = 0
                mx_pk = -1
                for i in range(opts.debug_search_step):
                    tmp = cut1 + cut2 * np.exp(1j*2*math.pi*i/opts.debug_search_step)
                    if np.max(np.abs(tmp)) > mx_pk:
                        mx_pk = np.max(abs(tmp))
                        out_rst = tmp
                        comp = 2*1j*i/opts.debug_search_step
                fft_raw=out_rst

                symbol_est = round(np.argmax(abs(fft_raw))/opts.debug_upsampling)%opts.n_classes
                if symbol_est == symbol_idx: ACC[symbol_idx] += 1
                SUM[symbol_idx] += 1
                pbar.set_description(f'{np.mean(ACC[SUM>0]/SUM[SUM>0]):.5f}')
    print('\rSNR:', opts.snr, 'ACC:', str(np.mean(ACC[SUM>0]/SUM[SUM>0])))
