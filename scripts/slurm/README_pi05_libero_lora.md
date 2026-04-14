# pi0.5 LIBERO LoRA and SWAG-LoRA on the ECE Cluster

This directory contains the cluster launchers for paper-style LoRA training on
`pi05_libero` and the matching SWAG-LoRA variant.

## What to run

Plain LoRA baseline:

```bash
sbatch scripts/slurm/sbatch_pi05_libero_lora_h100.sh
```

SWAG-LoRA:

```bash
sbatch scripts/slurm/sbatch_pi05_libero_swag_lora_h100.sh
```

## Common overrides

Set an explicit experiment name and seed:

```bash
sbatch --export=ALL,EXP_NAME=pi05_libero_lora_seed3,SEED=3 \
  scripts/slurm/sbatch_pi05_libero_lora_h100.sh
```

```bash
sbatch --export=ALL,EXP_NAME=pi05_libero_swag_lora_seed3,SEED=3 \
  scripts/slurm/sbatch_pi05_libero_swag_lora_h100.sh
```

Resume a run:

```bash
sbatch --export=ALL,EXP_NAME=pi05_libero_lora_seed3,SEED=3,RESUME=1 \
  scripts/slurm/sbatch_pi05_libero_lora_h100.sh
```

```bash
sbatch --export=ALL,EXP_NAME=pi05_libero_swag_lora_seed3,SEED=3,RESUME=1 \
  scripts/slurm/sbatch_pi05_libero_swag_lora_h100.sh
```

Tune SWAG collection:

```bash
sbatch --export=ALL,EXP_NAME=pi05_libero_swag_lora_seed3,SEED=3,SWAG_START_STEP=20000,SWAG_COLLECT_INTERVAL=1000,SWAG_MAX_NUM_MODELS=10 \
  scripts/slurm/sbatch_pi05_libero_swag_lora_h100.sh
```

## What each job does

`sbatch_pi05_libero_lora_h100.sh`

- prepares a job-local `uv` environment on scratch
- reuses persistent Hugging Face and `uv` caches
- computes LIBERO norm stats if they are missing
- launches `scripts/train_pi05_libero_lora.py`

`sbatch_pi05_libero_swag_lora_h100.sh`

- does the same environment setup
- launches `scripts/train_pi05_libero_swag_lora.py`
- saves normal checkpoints plus:
  - `swag_lora_state.pkl`
  - `swag_lora_summary.json`

## Output locations

By default, persistent outputs go under:

- checkpoints: `${HOME}/openpi-runs/checkpoints`
- wandb: `${HOME}/openpi-runs/wandb`

For the LoRA baseline, the checkpoint directory is:

```text
${HOME}/openpi-runs/checkpoints/pi05_libero_lora/<EXP_NAME>
```

For SWAG-LoRA, the checkpoint directory is also:

```text
${HOME}/openpi-runs/checkpoints/pi05_libero_lora/<EXP_NAME>
```

with the extra SWAG files saved inside that experiment directory.
