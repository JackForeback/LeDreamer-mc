#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=170G
#SBATCH --gpus-per-node=tesla_v100s:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=SUBSET
#SBATCH --output=~/LeDreamer-mc/outputs/ml-job%j.out
#SBATCH --error=~/LeDreamer-mc/errors/ml-job%j.err
#SBATCH --mail-user=email@gmail.com
##SBATCH --mail-type=end,fail

source ~/LeDreamer-mc/train-env/bin/activate

# V100S is Volta: fp16 via Tensor Cores, no bf16 support.
# --num_workers 8 matches --cpus-per-task above so frame decoding overlaps
# GPU compute.
# --resume_from latest is a no-op on the first run; on subsequent Slurm jobs
# it picks up from the newest accelerator.save_state() dump automatically.

# Phase 1: Train tokenizer (compresses video to latents)
python ~/train_dreamer4_minecraft.py --phase 1 \
    --data_dir ~/download/data \
    --output_dir ~/checkpoints \
    --num_steps 50000 \
    --num_workers 8 \
    --mixed_precision fp16 \
    --resume_from latest

# Phase 2: Train world model (learns dynamics from latents + actions)
python ~/train_dreamer4_minecraft.py --phase 2 --use_lewm \
    --data_dir ~/download/data \
    --output_dir ~/checkpoints \
    --tokenizer_ckpt ~/checkpoints/tokenizer.pt \
    --num_steps 100000 \
    --num_workers 8 \
    --mixed_precision fp16 \
    --resume_from latest

# Phase 3: Train agent in dreams (no real environment needed)
python ~/train_dreamer4_minecraft.py --phase 3 --use_lewm \
    --output_dir ~/checkpoints \
    --dynamics_ckpt ~/checkpoints/lewm_dynamics.pt \
    --num_steps 50000 \
    --mixed_precision fp16 \
    --resume_from latest

deactivate

source ~/LeDreamer-mc/eval-env/bin/activate

# 6. Evaluate in MineRL
python evaluate_dreamer4_minecraft.py \
    --checkpoint ~/checkpoints/dreamer4_minecraft.pt \
    --n_episodes 100 \
    --n_workers 8

deactivate

