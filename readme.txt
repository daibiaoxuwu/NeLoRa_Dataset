This folder provides the neural-based LoRa demodulation code for our ICLR 2023 workshop paper: NELORA-BENCH: A BENCHMARK FOR NEURAL ENHANCED LORA DEMODULATION
If the dataset and checkpoint are not downloadable due to git LFS bandwidth usage, they can be accessed at https://drive.google.com/drive/folders/12o3kqfBGrWG2YWegBa-sqErpAUsmLIOO
usage:
1. download NeLoRa_Dataset.zip from https://drive.google.com/drive/folders/12o3kqfBGrWG2YWegBa-sqErpAUsmLIOO and unzip into a certain location(e.g. /path/to/dataset/, containing 4 folders, 7, 8, 9, 10);
2. put the checkpoint files (70000_C_XtoY.pkl, 70000_maskCNN.pkl) at a certain location(e.g. /path/to/checkpoint_SF8/)
3. run: (using snr=-18, sf=8 for example)
python3 main.py --train_iters 0 --snr -18 --sf 8 --lr 0.0001 --data_dir /path/to/dataset/8 --batch_size 8 --w_image 1024 --checkpoint_dir /path/to/checkpoint_SF8 --test_step 100

typical output would be:

================================================================================
                                      Opts
--------------------------------------------------------------------------------
COMMAND:     main.py --train_iters 0 --snr -18 --sf 8 --lr 0.0001 --data_dir /path/to/NeLoRa_Dataset/NeLoRa_Dataset/8 --batch_size 8 --w_image 1024 --checkpoint_dir /path/to/checkpoints --max_test_iters 100
opts.conv_dim_lstm  2048
LOAD ITER:   70000
LOAD MODEL: /path/to/checkpoints/fver_0121_M0/70000_maskCNN.pkl
Current Time = 2023-02-05 00:36:48
 data_dir              /path/to/NeLoRa_Dataset/NeLoRa_Dataset/8
 sf                    8
 snr                   -18.0
 batch_size            8
 lr                    0.0001
 w_image               1024.0
 checkpoint_dir        /path/to/checkpoints/fver_0121_M0
 load_checkpoint_dir   /path/to/checkpoints/fver_0121_M0
 load                  yes
 load_iters            70000
 dechirp               True
================================================================================
read data: max cnt 524 0 min cnt 1 102
   CURRENT TIME       ITER  YLOSS  ILOSS  CLOSS   ACC   TIME  ----TRAINING 0.0001 ----
SAVED TEST SAMPLE: /path/to/checkpoints/sample-070001-snr-18.0-Yval.png
TEST: ACC: 0.9824999570846558 [98.25/100] ILOSS:  0.397 CLOSS:  0.092
REACHED 0.85 ACC, TERMINATINg...

4. run the baseline method (the dechirp method):
python3 main_baseline.py --snr -18 --sf 8 --data_dir /path/to/dataset/8 --rep 1
typical behavior:
0.9074149741766075: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 4172/4172 [05:26<00:00, 12.76it/s] SNR: -18.0 ACC: 0.9074149741766075

Additional Notes:
the filenames are:  {symbol_position_in_packet}_{code}_{packet_index}_{spreading_factor}.mat
Its format is a binary array of complex64 numbers (two consecutive float32 for real and imag part, a total of 64 bits per number) and can be read by (in python, for example) opening as binary and reading with np.fromfile(fid, np.complex64)
In the paper NELoRa: Towards Ultra-low SNR LoRa Communication with Neural-enhanced Demodulation, a cross-environment evaluation is performed where data is collected at 6 different locations before evaluated. The SF7 data in this dataset is organized into 'ins1' to 'ins6' subfolders. To evaluate performance at the 1st environment, simply specify data_dir to /path/to/dataset/7/ins1 instead of /path/to/dataset/7.

For any questions welcome to contact dujluo@gmail.com