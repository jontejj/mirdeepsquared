#!/bin/bash

#SBATCH -A naiss2023-22-767
#SBATCH -p core -n 16
#SBATCH -t 40:00:00
#SBATCH -J gunzip_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL


gunzip *_trimmed.fastq.gz
