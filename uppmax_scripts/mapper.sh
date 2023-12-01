#!/bin/sh

#SBATCH -A naiss2023-22-767
#SBATCH -p node -n 1
#SBATCH -t 50:00:00
#SBATCH -J build_mapper_files_mmu_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL

module load bioinfo-tools
module load mirdeep2

mapper.pl config.txt -d -e -m -p GRCm39 -s reads_collapsed.fa -t reads_collapsed_vs_genome.arf -v -h

