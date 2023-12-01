#!/bin/bash

#SBATCH -A naiss2023-22-767
#SBATCH -p core -n 16
#SBATCH -t 20:00:00
#SBATCH -J mirtrace_trimmed1_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL

#cd $1
ls *trimmed.fastq.gz > mirtrace_config_trimmed
/proj/snic2022-23-168/programs/mirtrace/mirtrace qc --species mmu --num-threads 16 --config mirtrace_config_trimmed --output-dir miRTrace_QC_trimmed_correctly 
