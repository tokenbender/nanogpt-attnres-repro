# FineWeb10B with Full Attention Residuals
# ~20M param GPT-2 style model
#
# Usage:
#   python train.py config/train_fineweb10B_attnres_full.py
#   torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_attnres_full.py

out_dir = "out-fineweb10B-attnres-full"
wandb_run_name = "attnres-full"

dataset = "fineweb10B"

# model
block_size = 1024
n_layer = 6
n_head = 6
n_embd = 288
dropout = 0.0
bias = False

batch_size = 32
gradient_accumulation_steps = 4
max_iters = 5000
eval_interval = 500
log_interval = 10
eval_iters = 100

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# residual mechanism
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True
mhc = False
attnres_variant = "full"
