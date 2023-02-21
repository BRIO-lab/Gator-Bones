#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd

module load conda


conda activate gatorBones

# Run a tutorial python script within the container. Modify the path to your container and your script.
python scripts/fit.py config
