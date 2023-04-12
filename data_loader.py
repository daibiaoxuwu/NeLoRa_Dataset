import os 
import random 
import numpy as np 
import torch 
from torch.utils.data import DataLoader 
from torch.utils import data 
from utils import *
import cv2 
import math 
from scipy.signal import chirp
from scipy.ndimage.filters import uniform_filter1d 
 
class lora_dataset(data.Dataset): 
    'Characterizes a dataset for PyTorch' 
 
    def __init__(self, opts, files): 
        self.opts = opts 
        self.files = files 
 
    def __len__(self): 
        return np.iinfo(np.int64).max 
 
    def load_img(self, path): 
        fid = open(path, 'rb') 
        nelements = self.opts.n_classes * self.opts.fs // self.opts.bw 
        lora_img = np.fromfile(fid, np.float32, nelements * 2) 
        lora_img = lora_img[::2] + lora_img[1::2]*1j 
        return torch.tensor(lora_img) 
 
 
    def __getitem__(self, index0): 
            try: 
                symbol_index = random.randint(0,self.opts.n_classes-1) 
                while(len(self.files[symbol_index]) < 1):  
                    symbol_index = random.randint(0,self.opts.n_classes-1) 

                data_perY = self.load_img(random.choice(self.files[symbol_index])).cuda() 
 
                data_pers = [] 

                #generate base downchirp
                nsamp = int(self.opts.fs * self.opts.n_classes / self.opts.bw) 
                t = np.linspace(0, nsamp / self.opts.fs, nsamp) 
                chirpI1 = chirp(t, f0=self.opts.bw/2, f1=-self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw , method='linear', phi=90) 
                chirpQ1 = chirp(t, f0=self.opts.bw/2, f1=-self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw, method='linear', phi=0) 
                chirp_down = chirpI1+1j*chirpQ1 
                 
                chirp_raw = data_perY

                #augumentation: randomly shift phase
                phase = random.uniform(-np.pi, np.pi) 
                chirp_raw *= (np.cos(phase)+1j*np.sin(phase)) 

                #gen ideal symbol
                nsamp = int(self.opts.fs * self.opts.n_classes / self.opts.bw) 
                t = np.linspace(0, nsamp / self.opts.fs, nsamp) 
                chirpI1 = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw , method='linear', phi=90) 
                chirpQ1 = chirp(t, f0=-self.opts.bw/2, f1=self.opts.bw/2, t1=2** self.opts.sf / self.opts.bw, method='linear', phi=0) 
                mchirp = chirpI1+1j*chirpQ1 
                mchirp = np.tile(mchirp, 2)
                time_shift = round(symbol_index / self.opts.n_classes * nsamp)
                chirp_ideal = torch.tensor(mchirp[time_shift:time_shift+nsamp],dtype=torch.cfloat)

                images_X_spectrum_ideal = torch.stft(input=chirp_ideal,n_fft=self.opts.stft_nfft,win_length =self.opts.stft_nfft//self.opts.stft_mod, hop_length =int(self.opts.stft_nfft/32), return_complex=True).cuda()
                ideal_symbol = torch.squeeze(spec_to_network_input2( spec_to_network_input(images_X_spectrum_ideal.unsqueeze(0), self.opts), self.opts )).cpu()

                images_X_spectrum_raw = torch.stft(input=chirp_raw,n_fft=self.opts.stft_nfft,win_length =self.opts.stft_nfft//self.opts.stft_mod, hop_length =int(self.opts.stft_nfft/32), return_complex=True).cuda()
                fake_symbol = torch.squeeze(spec_to_network_input2( spec_to_network_input(images_X_spectrum_raw.unsqueeze(0), self.opts), self.opts )).cpu()
                
                loss = torch.nn.MSELoss(reduction='mean')(torch.abs(ideal_symbol[0]+1j*ideal_symbol[1]), torch.abs(fake_symbol[0]+1j*fake_symbol[1]))
                if loss>0.00025:
                    #this symbol is too noisy
                    return self.__getitem__(index0) 

                mwin = nsamp//2; 
                datain = to_data(chirp_raw) 
                A = uniform_filter1d(abs(datain),size=mwin) 
                datain = datain[A >= max(A)/2] 
                amp_sig = torch.mean(torch.abs(torch.tensor(datain))) 
                chirp_raw /= amp_sig #normalization
                 
                amp = math.pow(0.1, self.opts.snr/20) 
                nsamp = self.opts.n_classes * self.opts.fs // self.opts.bw 
                noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat).cuda() 

                data_per = (chirp_raw + noise).cuda()
                label_per = torch.tensor(symbol_index, dtype=int).cuda() 
 
                return data_per, label_per, data_perY
            except ValueError as e: 
                print(e) 
            except OSError as e: 
                print(e) 
 

def lora_loader(opts): 

    #read filelist
    files = dict(zip(list(range(opts.n_classes)), [[] for i in range(opts.n_classes)])) 
    for subfolder in os.listdir(opts.data_dir):
        for filename in os.listdir(os.path.join(opts.data_dir, subfolder)):
            symbol_idx = int(filename.split('_')[1]) % opts.n_classes 
            files[symbol_idx].append(os.path.join(opts.data_dir, subfolder, filename))

    #split 8:2
    for i in files.keys(): 
        files[i].sort(key = lambda x: hash(os.path.basename(x)))
    splitpos = [int(len(files[i]) * opts.ratio_bt_train_and_test) for i in range(opts.n_classes)] 

    #debug
    a = [len(files[i]) for i in range(opts.n_classes)] 
    print('read data: max cnt', max(a), a.index(max(a)), 'min cnt', min(a), a.index(min(a)) )
 
    training_dataset = lora_dataset(opts, dict(zip(list(range(opts.n_classes)), [files[i][:splitpos[i]] for i in range(opts.n_classes)])) ) 
    training_dloader = DataLoader(dataset=training_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True) 
    testing_dataset = lora_dataset(opts, dict(zip(list(range(opts.n_classes)), [files[i][splitpos[i]:] for i in range(opts.n_classes)])) ) 
    testing_dloader = DataLoader(dataset=testing_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True) 
    return training_dloader, testing_dloader 
 
 
 
 

