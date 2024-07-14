import os
import math
import random
import argparse 

import torch
import numpy as np
from scipy.signal import chirp
from scipy.fft import fft
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from model_components import maskCNNModel, classificationHybridModel

np.random.seed(10)
random.seed(10)

# parameters
parser = argparse.ArgumentParser() 
parser.add_argument('--sf', type=int, help='The spreading factor.') 
parser.add_argument('--batch_size', type=int, default=16, help='The batch size.') 
opts = parser.parse_args()
sf = opts.sf  # spreading factor
batch_size = opts.batch_size  # batch size (the larger, the better, depending on GPU memory)

bw = 125e3  # bandwidth
fs = 1e6  # sampling frequency
data_dir = f'/path/to/NeLoRa_Dataset/{sf}/'  # directory for training dataset
mask_CNN_load_path = f'checkpoint/sf{sf}/100000_maskCNN.pkl'  # path for loading mask_CNN model weights
C_XtoY_load_path = f'checkpoint/sf{sf}/100000_C_XtoY.pkl'  # path for loading mask_CNN model weights
save_ckpt_dir = f'ckpt_sf{sf}'  # directory for saving trained weight checkpoints
normalization = True  # whether to perform normalization on data
snr_range = list(range(-30, 1))  # range of SNR for training
test_snr = -17  # SNR for testing
scaling_for_imaging_loss = 128  # scaling of losses between mask_CNN and C_XtoY
train_epochs = 100  # how many epochs to train (the larger, the better, network will not overfit)

# make directory for saving trained weight checkpoints
if not os.path.exists(save_ckpt_dir):
    os.mkdir(save_ckpt_dir)

# constants
num_classes = 2 ** sf  # number of codes per symbol == 2 ** sf
num_samples = int(num_classes * fs / bw)  # number of samples per symbol

# define models
mask_CNN = maskCNNModel(conv_dim_lstm=num_samples, lstm_dim=400, fc1_dim=600, freq_size=num_classes)
C_XtoY = classificationHybridModel(conv_dim_in=2, conv_dim_out=num_classes, conv_dim_lstm=num_samples)

# load models (remove if train from scratch)
mask_CNN.load_state_dict(torch.load(mask_CNN_load_path, map_location=lambda storage, loc: storage), strict=True)
C_XtoY.load_state_dict(torch.load(C_XtoY_load_path, map_location=lambda storage, loc: storage), strict=True)

# load models to GPU
mask_CNN.cuda()
C_XtoY.cuda()

# generate downchirp
t = np.linspace(0, num_samples / fs, num_samples + 1)[:-1]
chirpI1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=90)
chirpQ1 = chirp(t, f0=bw / 2, f1=-bw / 2, t1=2 ** sf / bw, method='linear', phi=0)
downchirp = chirpI1 + 1j * chirpQ1


# decoding symbols using loraphy, as baseline method
# note: this method only works with upsampling (FS >= BW*2)
def decode_loraphy(data_in, num_classes, downchirp):
    upsampling = 100  # up-sampling rate for loraphy, default 100
    # upsamping can counter possible frequency misalignments, finding the highest position of the signal peak, but higher upsampling lead to more noise

    # dechirp
    chirp_data = data_in * downchirp

    # compute FFT
    fft_raw = fft(chirp_data, len(chirp_data) * upsampling)

    # cut the FFt results to two (due to upsampling)
    target_nfft = num_classes * upsampling
    cut1 = np.array(fft_raw[:target_nfft])
    cut2 = np.array(fft_raw[-target_nfft:])

    # add absolute values of cut1 and cut2 to merge two peaks into one
    return round(np.argmax(abs(cut1) + abs(cut2)) / upsampling) % num_classes


# decoding symbols using our model for training and testing
def perform_stft(data_in):
    stft_full_img = torch.stft(input=data_in, n_fft=num_samples,
                               hop_length=num_classes // 4, win_length=num_classes // 2, pad_mode='constant',
                               return_complex=True)

    # up-down concatenation, to remove blank spaces due to oversampling
    # (the image is originally fs bandwidth (height: num_samples), we only need bw bandwidth (height: num_classes))
    stft_img = torch.concat((stft_full_img[:, -num_classes // 2:, :], stft_full_img[:, 0:num_classes // 2, :]), axis=1)

    # complex numbers -> 2 channels of real numbers
    return torch.stack((stft_img.real, stft_img.imag), 1)  # y.shape: batch_size, 2, height(num_classes), width


def decode_model(input_img):
    # run mask_CNN to generate a masked image
    mask_Y = mask_CNN(input_img.cuda())

    # classification
    outputs = C_XtoY(mask_Y)

    # return masked image, the prediction output, and the stft image
    return mask_Y, outputs


# adding noise for data
def add_noise(data_in, snr):
    # load data. dataY: data without noise
    dataY, truth_idx = data_in
    # add noise of a certain SNR, chosen from snr_range
    amp = math.pow(0.1, snr / 20) * torch.mean(torch.abs(dataY))
    noise = (amp / math.sqrt(2) * np.random.randn(num_samples) + 1j * amp / math.sqrt(2) * np.random.randn(
        num_samples)).type(torch.cfloat)
    dataX = dataY + noise  # dataX: data with noise
    if normalization:
        dataX = dataX / torch.mean(torch.abs(dataX)) # normalization
    return dataX, dataY, truth_idx

# load the whole dataset
def load_data():

    # cache read file results for faster start
    if os.path.exists(f'pkl_{sf}.pkl'):
        with open(f'pkl_{sf}.pkl', 'rb') as g:
            datax, datay = pickle.load(g)
        return datax, datay
        
    # read all file paths
    files = [[] for i in range(num_classes)]
    for subfolder in os.listdir(data_dir):
        for filename in os.listdir(os.path.join(data_dir, subfolder)):
            truth_idx = int(filename.split('_')[1])
            files[truth_idx].append(os.path.join(data_dir, subfolder, filename))

    # read file contents
    datax = []  # chirp symbols
    datay = []  # truth indexes for each symbol
    for truth_idx, filelist in tqdm(enumerate(files), desc = 'Reading Files'):
        for filepath in filelist:
            with open(filepath, 'rb') as fid:
                # read file
                chirp_raw = np.fromfile(fid, np.complex64, num_samples)
                assert len(chirp_raw) == num_samples
                # check if code is correct
                if decode_loraphy(chirp_raw, num_classes, downchirp) == truth_idx:
                    # append data
                    datax.append(torch.tensor(chirp_raw, dtype=torch.cfloat))
                    datay.append(truth_idx)
    # cache read file results for faster start
    with open(f'pkl_{sf}.pkl', 'wb') as g:
        pickle.dump((datax, datay), g)

    return datax, datay


# train main function
def train():

    # create TensorDataset
    datax, datay = load_data()
    data = TensorDataset(torch.stack(datax), torch.tensor(datay, dtype=torch.long))
   
# Calculate class weights for imbalanced datasets
    class_counts = torch.bincount(torch.tensor(datay))
    class_weights = 1. / class_counts.float()
    sample_weights = class_weights[torch.tensor(datay)]

# Split the dataset into 9:1 ratio
    train_size = int(0.9 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = random_split(data, [train_size, test_size])

# Create samplers for weighted sampling
    train_weights = sample_weights[train_dataset.indices]
    test_weights = sample_weights[test_dataset.indices]

    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
    test_sampler = WeightedRandomSampler(weights=test_weights, num_samples=len(test_weights), replacement=True)

# Create DataLoaders
    training_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    testing_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    # initiate training
    g_params = list(mask_CNN.parameters()) + list(C_XtoY.parameters())
    g_optimizer = torch.optim.Adam(g_params, 0.0001, [0.5, 0.999])
    loss_spec = torch.nn.MSELoss(reduction='mean')
    loss_class = torch.nn.CrossEntropyLoss()

    # main training loop
    for epoch in range(train_epochs):
        for data_train in tqdm(training_loader):

            dataX, dataY, truth_idx = add_noise(data_train, random.choice(snr_range))

            # perform stft
            input_img = perform_stft(dataX)
            truth_img = perform_stft(dataY)

            # decode dataX with model
            output_img, est_train = decode_model(input_img)

            # compute loss
            g_y_pix_loss = loss_spec(output_img, truth_img.cuda())
            g_y_class_loss = loss_class(est_train, truth_idx.cuda())

            # add up loss with scaling_for_imaging_loss, and back porpagation
            g_optimizer.zero_grad()
            G_Y_loss = scaling_for_imaging_loss * g_y_pix_loss + g_y_class_loss
            G_Y_loss.backward()
            g_optimizer.step()

        # checkpoint
        torch.save(mask_CNN.state_dict(), os.path.join(save_ckpt_dir, str(epoch) + '_maskCNN.pkl'))
        torch.save(C_XtoY.state_dict(), os.path.join(save_ckpt_dir, str(epoch) + '_C_XtoY.pkl'))

        # test
        with torch.no_grad():
            mask_CNN.eval()
            C_XtoY.eval()

            # correct count
            correct_count = 0
            for data_test in testing_loader:
                dataX_test, dataY_test, truth_test = add_noise(data_test, test_snr)

                # perform stft
                input_img = perform_stft(dataX_test)

                # run model testing
                _, est_test = decode_model(input_img)

                est_code = torch.max(est_test, 1)[1].cpu()
                correct_count += torch.sum(est_code == truth_test)

            print('SNR: %d TEST ACC: %.3f' % (test_snr, correct_count / (len(testing_loader) * batch_size)))
            mask_CNN.train()
            C_XtoY.train()

# Test main function
def test():
    acc = np.zeros((2, len(snr_range),num_classes,))
    cnt = np.zeros((len(snr_range),num_classes,))
    datax, datay = load_data()
    data = TensorDataset(torch.stack(datax), torch.tensor(datay, dtype=torch.long))
    all_dataloader = DataLoader(data, batch_size=batch_size)
    for data_test in tqdm(all_dataloader):
        for snridx, snr in enumerate(snr_range):

            dataX_test, dataY_test, truth_test = add_noise(data_test, snr)

            est_loraphy = torch.tensor([decode_loraphy(dataX, num_classes, downchirp) for dataX in dataX_test.numpy()])
            input_img = perform_stft(dataX_test)
            _, est_test = decode_model(input_img)
            est_code = torch.max(est_test, 1)[1].cpu()
            acc[0][snridx][truth_test] += torch.sum(est_loraphy == truth_test).item()
            acc[1][snridx][truth_test] += torch.sum(est_code == truth_test).item()
            cnt[snridx][truth_test] += truth_test.shape[0]

    # save testing data
    with open(f'test_sf{sf}.pkl', 'wb') as g:
        pickle.dump((acc, cnt), g)

    # Plot
    plt.axhline(y=0.9, linestyle='--', color='black')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')

    for pidx, label in enumerate(['LoRaPHY', 'NeLoRa']):
        with np.errstate(divide='ignore', invalid='ignore'):
            div_result = np.where(cnt != 0, acc[pidx] / cnt, np.nan)
        res = np.nanmean(div_result, axis=1)
        plt.plot(snr_range, res, label=label)

    plt.legend()
    plt.savefig(f'test_sf{sf}.pdf')
    plt.clf()
    
    
if __name__ == '__main__':
    train()  # For training
    # test()  # For testing    





