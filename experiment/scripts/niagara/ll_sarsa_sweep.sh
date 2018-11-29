#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=2:59:00
#SBATCH --job-name 1_lunar_lander_sweep
#SBATCH --output=out_1_lunar_lander_sweep_%j.txt
#SBATCH --error=err_1_lunar_lander_sweep_%j.txt

cd $SLURM_SUBMIT_DIR/../../../
export OMP_NUM_THREADS=1
module load singularity/2.5.2


singularity exec -B /scratch/w/whitem/mzaheer/DeepRL cf-env.img parallel --jobs 75 --results /scratch/w/whitem/mzaheer/DeepRL/data/parallel_output/lunar_lander/sarsa python /scratch/w/whitem/mzaheer/DeepRL/run.py --id {1} --config-file experiment/config_files/lunar_lander/sarsa/sweep.json ::: $(seq 0 149)