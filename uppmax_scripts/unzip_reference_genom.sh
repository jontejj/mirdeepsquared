#!/bin/sh

#SBATCH -A naiss2023-22-767
#SBATCH -p node -n 1
#SBATCH -t 8:00:00
#SBATCH -J unzip_genome_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL

unzip GCF_000001635.27.zip
