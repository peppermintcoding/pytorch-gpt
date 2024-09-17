"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
log_interval = 10
checkpoint_step = 10_000
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# data
grad_acc_steps = 1  # used to simulate larger batch sizes
batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_steps = 10_000  # total number of training steps
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_steps = 100  # how many steps to warm up for
lr_decay_steps = max_steps * 0.95  # should be ~= max_steps per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"
dtype = torch.bfloat16
compile = True
plotting = True

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation steps per process proportionally
    assert grad_acc_steps % ddp_world_size == 0
    grad_acc_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_step = grad_acc_steps * ddp_world_size * batch_size * block_size
print(f"tokens per step will be: {tokens_per_step:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# poor man's data loader
data = np.load("train.npy")
print(f"total number of tokens: {len(data):,}")


def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = [(data[i : i + block_size]).astype(np.int64) for i in ix]
    y = [(data[i + 1 : i + 1 + block_size]).astype(np.int64) for i in ix]
    x = torch.stack([torch.from_numpy(xi) for xi in x])
    y = torch.stack([torch.from_numpy(yi) for yi in y])
    # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    x = x.pin_memory().to(device, non_blocking=True)
    y = y.pin_memory().to(device, non_blocking=True)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
step = 0
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    mlp_hidden_size=2*n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=50304,
    dropout=dropout,
)
if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = GPT(GPTConfig(**model_args))
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    step = checkpoint["step"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = (
        block_size  # so that the checkpoint will have the right value
    )
model.to(device)

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

if compile:
    print("compiling the model... (takes a ~minute)")
    # max-autotune does not work on a2000
    # reduce-overhead is slower
    # max-autotune-no-cudagraphs same as default for speed
    model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# learning rate decay scheduler (cosine with warmup)
def get_lr(step):
    # 1) linear warmup for warmup_steps
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    # 2) if it > lr_decay_steps, return min learning rate
    if step > lr_decay_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
start_time = time.time()
X, Y = get_batch()  # fetch the very first batch
raw_model = model.module if ddp else model  # unwrap DDP container if needed
loss_history = []
while step < max_steps:
    t0 = time.time()
    # determine and set the learning rate for this step
    lr = get_lr(step) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if step > 0 and step % checkpoint_step == 0:
        checkpoint = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "step": step,
        }
        print(f"saving checkpoint to {out_dir}")
        torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    for micro_step in range(grad_acc_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == grad_acc_steps - 1
        with torch.amp.autocast(device_type=device, dtype=dtype):
            logits, loss = model(X, Y)
            loss = loss / grad_acc_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch()
        loss.backward()
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    if step % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item() * grad_acc_steps
        loss_history.append((step, lossf))
        t1 = time.time()
        dt = t1 - t0
        remaining_time = int((max_steps - step) / log_interval * dt) / 60
        toks_per_second = int((tokens_per_step * grad_acc_steps * log_interval) / dt)
        print(
            f"step {step}: loss: {lossf:.4f} tok/s: {toks_per_second:,} remaining time: {remaining_time:.1f} minutes"
        )
    step += 1

checkpoint = {
    "model": raw_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "model_args": model_args,
    "step_num": step,
}
print(f"saving checkpoint to {out_dir}")
torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
print(f"training took {(time.time() - start_time) / 60:.2f} minutes")

if plotting:
    plt.plot([x[0] for x in loss_history], [x[1] for x in loss_history])
    plt.show()

if ddp:
    destroy_process_group()
