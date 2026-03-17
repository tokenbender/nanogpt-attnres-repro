# FineWeb10B with mHC (4 streams, 48 layers)
# ~20M param GPT-2 style model with higher depth for fuller multi-GPU utilization
#
# Usage:
#   python train.py config/train_fineweb10B_mhc_48l.py
#   torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B_mhc_48l.py

out_dir = "out-fineweb10B-mhc-48l"
wandb_run_name = "mhc-48l"

dataset = "fineweb10B"

# model
block_size = 1024
n_layer = 48
n_head = 6
n_embd = 150
dropout = 0.0
bias = False

# training
# Lock semantics to a 128-sequence global batch (131,072 tokens/update) and
# ~1.05B total tokens. Override `batch_size` per hardware; the trainer will
# derive accumulation to preserve the semantic batch.
batch_size = 8
gradient_accumulation_steps = 16
target_tokens_per_iter = 131072
target_tokens = 1048576000
max_iters = 8000
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
lr_decay_iters = 8000
lock_lr_decay_to_max_iters = True
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# hyper-connections: mHC enabled (4 streams)
hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False
mhc = True
sinkhorn_iters = 20
sinkhorn_tau = 0.05
mhc_h_res_proj = "sinkhorn"
ns_steps = 5
ns_eps = 1e-7
ns_coeffs = (3.0, -3.2, 1.2)
