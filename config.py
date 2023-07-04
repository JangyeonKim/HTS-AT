exp_name = "trial_pl" 
workspace = "/home/jykim/work/HTS-AT" # the folder of your code
dataset_path = "/home/jykim/work/HTS-AT/aihub_dataset/v2_MotorAndCar" # the dataset path

loss_type = "clip_bce" # "clip_bce" or "clip_ce" 

# swin_pretrain_path = None # pretrained weight path?
resume_checkpoint = "/home/jykim/work/HTS-AT/pretrained_weight/HTSAT_AudioSet_Saved_6.ckpt" # resume from checkpoint
test_checkpoint = None

device = [0]

random_seed = 970131 # 19970318 970131 12412 127777 1009 34047
batch_size = 64 # batch size per GPU x GPU number , default is 32 x 4 = 128
max_epoch = 100
num_workers = 3

learning_rate = 1e-3 # 1e-4 also workable 
lr_scheduler_epoch = [10,20,30]
lr_rate = [0.02, 0.05, 0.1]

enable_tscam = True # enbale the token-semantic layer

# for signal processing
sample_rate = 16000 # 16000 for scv2, 32000 for audioset and esc-50
clip_samples = sample_rate * 10 # audio_set 10-sec clip
window_size = 1024
hop_size = 160 # 160 for scv2, 320 for audioset and esc-50
mel_bins = 64
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)

# for data collection
classes_num = 3 # esc: 50 | audioset: 527 | scv2: 35
patch_size = (25, 4) # deprecated
crop_size = None # int(clip_samples * 0.5) deprecated

# for htsat hyperparamater
htsat_window_size = 8
htsat_spec_size =  256
htsat_patch_size = 4 
htsat_stride = (4, 4)
htsat_num_head = [4,8,16,32]
htsat_dim = 96 
htsat_depth = [2,2,6,2]

htsat_attn_heatmap = False
enable_repeat_mode = False
debug = False