#!/bin/bash
#SBATCH --mail-user=rishavdutta@umass.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cpus-per-gpu 12  # Number of Cores per Task
#SBATCH -p gpu
#SBATCH --mem-per-cpu=10GB
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o ./slurm/slurm-datasets-%j.out  # %j = job ID
#SBATCH --gres=gpu:2
#SBATCH --gpus=2
#SBATCH --constraint=a40|l40s|a100|h100
#SBATCH -A pi_hzhang2_umass_edu

# STAGE 1 training: synthetic data (HPatches)
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_homography_dino_vits8 --conf scripts/configs/superpoint+diffglue_homography.yaml encoder.extractor.modelname=dino_vits8 --run_benchmarks --restore
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_homography_dino_vits16 --conf scripts/configs/superpoint+diffglue_homography.yaml encoder.extractor.modelname=dino_vits16 --run_benchmarks --restore
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_homography_dinov2_vits14 --conf scripts/configs/superpoint+diffglue_homography.yaml encoder.extractor.modelname=dinov2_vits14 --run_benchmarks --restore


# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_homography_dino_vits8_scale1 --conf scripts/configs/superpoint+diffglue_homography.yaml encoder.extractor.modelname=dino_vits8 encoder.extractor.scale_factors=[1] --run_benchmarks --restore
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_homography_dino_vits16_scale1 --conf scripts/configs/superpoint+diffglue_homography.yaml encoder.extractor.modelname=dino_vits16 encoder.extractor.scale_factors=[1] --run_benchmarks --restore
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_homography_dinov2_vits14_scale1 --conf scripts/configs/superpoint+diffglue_homography.yaml encoder.extractor.modelname=dinov2_vits14 encoder.extractor.scale_factors=[1] --run_benchmarks --restore



# STAGE 2 training: real data (megadepth)
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_dino_vits8 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dino_vits8 data.load_features.do=True encoder.extractor.modelname=dino_vits8 --run_benchmarks
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_dino_vits16 --conf scripts/configs/superpoint+diffglue_megadepth.yaml data.load_features.do=True encoder.extractor.modelname=dino_vits16 --run_benchmarks --restore
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_dinov2_vits14 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dinov2_vits14 data.load_features.do=True encoder.extractor.modelname=dinov2_vits14 --run_benchmarks --distributed  --restore


# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_dino_vits8_scale1 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dino_vits8_scale1 data.load_features.do=True encoder.extractor.modelname=dino_vits8 encoder.extractor.scale_factors=[1] --run_benchmarks
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_dino_vits16_scale1 --conf scripts/configs/superpoint+diffglue_megadepth.yaml data.load_features.do=True encoder.extractor.modelname=dino_vits16 encoder.extractor.scale_factors=[1] --run_benchmarks --restore
apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_dinov2_vits14_scale1 --conf scripts/configs/superpoint+diffglue_megadepth.yaml data.load_features.do=True encoder.extractor.modelname=dinov2_vits14 encoder.extractor.scale_factors=[1] --run_benchmarks --distributed --restore


# STAGE 2 training: real data (megadepth-epipolar)
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_epipolar_dino_vits8 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dino_vits8 data.load_features.do=True encoder.extractor.modelname=dino_vits8 --run_benchmarks
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_epipolar_dino_vits16 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dino_vits16 data.load_features.do=True encoder.extractor.modelname=dino_vits16 --run_benchmarks
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_epipolar_dinov2_vits14 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dinov2_vits14 data.load_features.do=True encoder.extractor.modelname=dinov2_vits14 --run_benchmarks --distributed


# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_epipolar_dino_vits8_scale1 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dino_vits8_scale1 data.load_features.do=True encoder.extractor.modelname=dino_vits8 encoder.extractor.scale_factors=[1] --run_benchmarks
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_epipolar_dino_vits16_scale1 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dino_vits16_scale1 data.load_features.do=True encoder.extractor.modelname=dino_vits16 encoder.extractor.scale_factors=[1] --run_benchmarks
# apptainer exec --nv /project/pi_hzhang2_umass_edu/rishavdutta_umass_edu/git_repos/DiffGlue/diffglue.sif python3 -m scripts.train SP+DiffGlue_megadepth_epipolar_dinov2_vits14_scale1 --conf scripts/configs/superpoint+diffglue_megadepth.yaml train.load_experiment=SP+DiffGlue_homography_dinov2_vits14_scale1 data.load_features.do=True encoder.extractor.modelname=dinov2_vits14 encoder.extractor.scale_factors=[1] --run_benchmarks --distributed
