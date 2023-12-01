#!/bin/bash

#SBATCH -A naiss2023-22-767
#SBATCH -p core -n 16
#SBATCH -t 40:00:00
#SBATCH -J download_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL


module load bioinfo-tools
module load sratools

for file in `cat SRR_IDS.txt`
do
echo $file
fasterq-dump -e 16 -t temp $file
gzip ${file}.fastq
done
