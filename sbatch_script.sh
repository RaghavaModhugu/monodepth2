#!/bin/bash
#SBATCH --nodelist=gnode32
#SBATCH --gres=gpu:1
#SBATCH -n 12
#SBATCH --mem-per-cpu=2G
#SBATCH --time=INFINITE
#SBATCH -p long
#SBATCH --job-name="monodepth2"

module load use.own
module load cuda/10.0
module load cudnn/7-cuda-10.0

source activate monodepth2
wait

cd /home/raghava.modhugu/self_supervised_trajectory_prediction/monodepth2

wait

mkdir -p /ssd_scratch/cvit/raghava.modhugu
wait

mkdir -p /ssd_scratch/cvit/raghava.modhugu/tmp
wait

python3 train.py --model_name argoverse_mono --log_dir /ssd_scratch/cvit/raghava.modhugu/tmp --data_path /ssd_scratch/cvit/raghava.modhugu/
wait

