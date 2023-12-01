#!/bin/sh

#SBATCH -A naiss2023-22-767
#SBATCH -p node -n 1
#SBATCH -t 8:00:00
#SBATCH -J build_bowtie_index_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL

module load bioinfo-tools
module load bowtie

bowtie-build ncbi_dataset/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna GRCm39 --threads 8
