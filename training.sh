#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiayu.huang@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:3
#SBATCH --time=40:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load conda
conda activate JTML
# export PATH="/home/sasank.desaraju/conda_envs/envs/jtml/bin/:$PATH"
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/sasank.desaraju/conda_envs/envs/jtml/lib"


# Run a tutorial python script within the container. Modify the path to your container and your script.
python scripts/fit.py config
