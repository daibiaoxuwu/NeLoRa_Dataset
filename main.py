"""Main script for project."""
from __future__ import print_function
import config
import data_loader
import end2end
import os
import numpy as np
import torch
import sys
from model_components0 import maskCNNModel0, classificationHybridModel0
from utils import *
import collections

def load_checkpoint(opts, maskCNNModel, classificationHybridModel):
    maskCNN_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_maskCNN.pkl')

    C_XtoY_path = os.path.join(opts.load_checkpoint_dir, str(opts.load_iters) + '_C_XtoY.pkl')
    print('LOAD MODEL:', maskCNN_path)

    maskCNN = maskCNNModel(opts)
    if opts.load_maskcnn == 'True':
        state_dict = torch.load(maskCNN_path, map_location=lambda storage, loc: storage)
        for key in list(state_dict.keys()): state_dict[key.replace('module.', '')] = state_dict.pop(key)
        #state_dict['conv2.1.weight']= torch.cat((state_dict['conv2.1.weight'], torch.zeros(64,258-130,5,5)),1)
        #state_dict['fc1.weight']= state_dict['fc1.weight'][:, :4096]
        #state_dict.pop('fc2.weight')
        #state_dict.pop('fc2.bias')
        maskCNN.load_state_dict(state_dict)#, strict=False)

    if opts.cxtoy == 'True':
        C_XtoY = classificationHybridModel(conv_dim_in=opts.x_image_channel, conv_dim_out=opts.n_classes, conv_dim_lstm=opts.conv_dim_lstm)
        if opts.load_cxtoy == 'True' and os.path.exists(C_XtoY_path):
            state_dict = torch.load( C_XtoY_path, map_location=lambda storage, loc: storage)
            if type(state_dict)==collections.OrderedDict:
                for key in list(state_dict.keys()): state_dict[key.replace('module.', '')] = state_dict.pop(key)
                #state_dict['dense.weight']= state_dict['dense.weight'][:,:state_dict['dense.weight'].shape[1]//opts.stack_imgs ]
                C_XtoY.load_state_dict(state_dict)#, strict=False)
            else:
                C_XtoY = torch.load(C_XtoY_path)
                
        return [maskCNN, C_XtoY]
    else: return [maskCNN, ]


def main(opts,models):
    torch.cuda.empty_cache()

    # Create train and test dataloaders for images from the two domains X and Y
    training_dataloader, testing_dataloader = data_loader.lora_loader(opts)
    # Create checkpoint directories

    # Start training
    set_gpu(opts.free_gpu_id)

    # start training
    models = end2end.training_loop(training_dataloader,testing_dataloader,models, opts)
    return models

if __name__ == "__main__":
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    print('COMMAND:    ', ' '.join(sys.argv))
    parser = config.create_parser()
    opts = parser.parse_args()

    if opts.sf == -1:
        opts.sf = int(opts.checkpoint_dir.split('-')[-1])
        opts.data_dir='/data/djl/SF'+str(opts.sf)+'_125K'

    opts.n_classes = 2 ** opts.sf
    opts.stft_nfft = opts.n_classes * opts.fs // opts.bw

    opts.stft_window = opts.n_classes // 2 * opts.stft_mod
    opts.stft_overlap = opts.stft_window // 2 // opts.stft_mod
    opts.conv_dim_lstm = opts.n_classes * opts.fs // opts.bw
    print('opts.conv_dim_lstm ',opts.conv_dim_lstm )
    opts.freq_size = opts.n_classes

    create_dir(opts.checkpoint_dir)

    
    if opts.lr == -1:
        opts.lr = 0.001
        if min(opts.snr) < -15: opts.lr *= 0.3
        if min(opts.snr) < -20: opts.lr /= 1.5
    if opts.w_image == -1:
        opts.w_image = 1
        if min(opts.snr) < -15: opts.w_image *= 4
        if min(opts.snr) < -20: opts.w_image *= 4


    #default checkpoint dir
    if opts.load_checkpoint_dir == '/data/djl': opts.load_checkpoint_dir = opts.checkpoint_dir

    
    maskCNNModel = maskCNNModel0
    classificationHybridModel = classificationHybridModel0

    if opts.load == 'yes':
        if opts.load_iters == -1:
            vals = [int(fname.split('_')[0]) for fname in os.listdir(opts.load_checkpoint_dir) if fname[-4:] == '.pkl']
            if len(vals)==0 or max(vals) == 0: 
                opts.load = 'no'
                print('--WARNING: CHECKPOINT_DIR NOT EXIST, SETTING OPTS.LOAD TO NO--')
            else: opts.load_iters = max(vals)
    codepath = os.path.join(opts.checkpoint_dir, 'code'+str(opts.load_iters))
    create_dir(codepath)
    os.system('cp '+ os.path.dirname(os.path.abspath(__file__))+r'/*.py '+codepath)
    

    if opts.load == 'yes':
        print('LOAD ITER:  ',opts.load_iters)
        models = load_checkpoint(opts, maskCNNModel, classificationHybridModel)
        mask_CNN = models[0]
        if opts.cxtoy == 'True': C_XtoY = models[1]
    else:
        mask_CNN = maskCNNModel(opts)
        if opts.cxtoy == 'True': C_XtoY = classificationHybridModel(conv_dim_in=opts.x_image_channel, conv_dim_out=opts.n_classes, conv_dim_lstm= opts.conv_dim_lstm)
    #mask_CNN = nn.DataParallel(mask_CNN)
    mask_CNN.cuda()
    models = [mask_CNN, ]
    if opts.cxtoy == 'True':
        #C_XtoY = nn.DataParallel(C_XtoY)
        C_XtoY.cuda()
        models.append(C_XtoY)
    
    opts.logfile = os.path.join(opts.checkpoint_dir, 'logfile-djl-train.txt')
    opts.logfile2 = os.path.join(opts.checkpoint_dir, 'logfile2-djl-train.txt')
    strlist = print_opts(opts)
    with open(opts.logfile,'a') as f: f.write('\n'+' '.join(sys.argv))
    with open(opts.logfile,'a') as f: f.write('\n'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' ' +'\n'.join(strlist)+'\n')
    with open(opts.logfile2,'a') as f: f.write(str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' snr ' +str(opts.snr)+' : ')
    opts.init_train_iter = opts.load_iters
    models = main(opts,models)
    opts.init_train_iter += opts.train_iters

