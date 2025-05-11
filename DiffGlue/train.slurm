#!/bin/bash
#SBATCH --mail-user=rishavdutta@umass.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-gpu 8  # Number of Cores per Task
#SBATCH -p gpu
#SBATCH --mem-per-cpu=8GB
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o ./slurm/slurm-datasets-%j.out  # %j = job ID
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --constraint=a40|l40s|l4
#SBATCH -A pi_hzhang2_umass_edu

apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_homography --conf scripts/configs/superpoint+diffglue_homography.yaml --run_benchmarks --restore
