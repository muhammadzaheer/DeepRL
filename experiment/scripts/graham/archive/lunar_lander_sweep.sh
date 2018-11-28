#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:59:00
#SBATCH --mem=40000M
#SBATCH --job-name 1_lunar_lander_sweep
#SBATCH --output=out_1_lunar_lander_sweep_%j.txt
#SBATCH --error=err_1_lunar_lander_sweep_%j.txt

cd $SLURM_SUBMIT_DIR/../../../
export OMP_NUM_THREADS=1
module load singularity

singularity exec -B /scratch/mzaheer/DeepRL cf-env.img parallel --jobs 30 --results /scratch/mzaheer/DeepRL/data/parallel_output/lunar_lander/sarsa python /scratch/mzaheer/DeepRL/run.py --id {1} --config-file experiment/config_files/lunar_lander/sarsa/sweep.json ::: $(seq 300 389)
