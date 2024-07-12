# NELoRa-Bench
This folder provides the neural-based LoRa demodulation code for our ICLR 2023 workshop paper: [NELoRa-Bench: A Benchmark for Neural-enhanced LoRa Demodulation](https://doi.org/10.48550/arXiv.2305.01573)
If the dataset and checkpoint are not downloadable due to git LFS bandwidth usage, they can be accessed at this [Google Drive link](https://drive.google.com/drive/folders/12o3kqfBGrWG2YWegBa-sqErpAUsmLIOO)

This code reproduces the experiments in the SenSys '21 paper "[NELoRa: Towards Ultra-low SNR LoRa Communication with Neural-enhanced Demodulation](https://cse.msu.edu/~caozc/papers/sensys21-li.pdf)".

Differences from the original [code provided by NELoRa](https://github.com/hanqingguo/NELoRa-Sensys):
1. Now for both nelora and baseline train/test, no need for a separate stage of data-generation (adding artificial noise). Noise is added on-the-fly. This reduces overfitting issues and removes the need for additional harddisk space, also speeding up the process drastically.
2. Added data balancing.
3. Removed clutter.
4. Parameters are hardcoded.
5. Add a double check on the dataset for wrong codes.
6. Add comparison with baseline methods, using LoRaPhy from [From Demodulation to Decoding: Toward Complete LoRa PHY Understanding and Implementation](https://doi.org/10.1145/3546869)

Usage:
1. Download dataset and unzip them from [Google Drive](https://drive.google.com/drive/folders/12o3kqfBGrWG2YWegBa-sqErpAUsmLIOO).
2. Download checkpoints from the same [Google Drive](https://drive.google.com/drive/folders/12o3kqfBGrWG2YWegBa-sqErpAUsmLIOO). Note: these checkpoints are only trained for a limited amount of time and can be improved.
3. Adjust the parameters in main.py:
```python
# parameters
sf = 8  # spreading factor
bw = 125e3  # bandwidth
fs = 1e6  # sampling frequency
data_dir = f'/path/to/NeLoRa_Dataset/{sf}/'  # directory for training dataset
mask_CNN_load_path = f'checkpoint/sf{sf}/100000_maskCNN.pkl'  # path for loading mask_CNN model weights
C_XtoY_load_path = f'checkpoint/sf{sf}/100000_C_XtoY.pkl'  # path for loading C_XtoY model weights
save_ckpt_dir = 'ckpt'  # directory for saving trained weight checkpoints
normalization = False  # whether to perform normalization on data
snr_range = list(range(-30, 1))  # range of SNR for training
test_snr = -22  # SNR for testing
batch_size = 16  # batch size (the larger, the better, depending on GPU memory)
scaling_for_imaging_loss = 128  # scaling of losses between mask_CNN and C_XtoY
ckpt_per_iter = 1000  # checkpoint per iteration
train_epochs = 1  # how many epochs to train (the larger, the better, network will not overfit)
```
3. For training, call train(). For testing, call test().
```python
if __name__ == '__main__':
    train()
    # test()    
```
4. please consider to cite our paper if you use the code or data in your research project.
```bibtex
  @inproceedings{nelora2021sensys,
  	title={{NELoRa: Towards Ultra-low SNR LoRa Communication with Neural-enhanced Demodulation}},
  	author={Li, Chenning and Guo, Hanqing and Tong, Shuai and Zeng, Xiao and Cao, Zhichao and Zhang, Mi and Yan, Qiben and Xiao, Li and Wang, Jiliang and Liu, Yunhao},
    	booktitle={In Proceeding of ACM SenSys},
    	year={2021}
  }
```
