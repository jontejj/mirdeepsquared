#!/bin/sh

#SBATCH -A naiss2023-22-767
#SBATCH -p node -n 1
#SBATCH -t 8:00:00
#SBATCH -J mirdeep2_mmu_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL

module load bioinfo-tools
module load mirdeep2

export PERL5LIB=/sw/bioinfo/mirdeep2/2.0.1.2/rackham/lib/perl5/
miRDeep2.pl reads_collapsed.fa GCF_000001635.27_GRCm39_genomic_no_spaces.fna reads_collapsed_vs_genome.arf known-mature-sequences-mouse.fas known-mature-sequences-norwegian-rat.fas mmu-pre.fas -b -5 -t mmu 2>report5.log

