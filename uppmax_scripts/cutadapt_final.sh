#!/bin/bash

#SBATCH -A naiss2023-22-767
#SBATCH -p core -n 16
#SBATCH -t 20:00:00
#SBATCH -J cutadapt_jonatan
#SBATCH --begin=now
#SBATCH --mail-type=ALL

module load bioinfo-tools
module load cutadapt

#This was the adapter used before the real adapters were discovered by zgrep'ping the fastq.gz files
export ADAPTER="UGAGGUAGGAGGUUGUAUAGUU"

cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACA -o SRR25511174_trimmed.fastq.gz SRR25511174.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACA -o SRR25338387_trimmed.fastq.gz SRR25338387.fastq.gz
cutadapt -j 16 -e 0.1 -a AGTCGGAGGCCAAGCG -o SRR25205118_trimmed.fastq.gz SRR25205118.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCC -o SRR22739462_trimmed.fastq.gz SRR22739462.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACA -o SRR24949848_trimmed.fastq.gz SRR24949848.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACA -o SRR17652208_trimmed.fastq.gz SRR17652208.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCC -o SRR10240201_trimmed.fastq.gz SRR10240201.fastq.gz
cutadapt -j 16 -e 0.1 -a TGGAATTCTCGGGTGC -o SRR8494799_trimmed.fastq.gz SRR8494799.fastq.gz
cutadapt -j 16 -e 0.1 -a TGGAATTCTCGGGTGC -o SRR6793409_trimmed.fastq.gz SRR6793409.fastq.gz
cutadapt -j 16 -e 0.1 -a TGGAATTCTCGGGTGC -o SRR6787032_trimmed.fastq.gz SRR6787032.fastq.gz
cutadapt -j 16 -e 0.1 -a TTGTATAGTT -o SRR1551219_trimmed.fastq.gz SRR1551219.fastq.gz
cutadapt -j 16 -e 0.1 -a CGTATGCC -o SRR1551218_trimmed.fastq.gz SRR1551218.fastq.gz
cutadapt -j 16 -e 0.1 -a TCGTATGCCGTCT -o SRR1551220_trimmed.fastq.gz SRR1551220.fastq.gz
cutadapt -j 16 -e 0.1 -a AGTCGGAGGCCAAGCGGTCTTAGGA -o SRR25205161_trimmed.fastq.gz SRR25205161.fastq.gz
cutadapt -j 16 -e 0.1 -a AGTCGGAGGCCAAGCGGTCTTAGG -o SRR25205080_trimmed.fastq.gz SRR25205080.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCA -o SRR25338389_trimmed.fastq.gz SRR25338389.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACGTCTGAACT -o SRR25338388_trimmed.fastq.gz SRR25338388.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACGTCT -o SRR25511175_trimmed.fastq.gz SRR25511175.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACGTCT -o SRR25511182_trimmed.fastq.gz SRR25511182.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACGTCTGAA -o SRR24949847_trimmed.fastq.gz SRR24949847.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACG -o SRR24949826_trimmed.fastq.gz SRR24949826.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCCAAGGAAC -o SRR22739477_trimmed.fastq.gz SRR22739477.fastq.gz
cutadapt -j 16 -e 0.1 -a TGGAATTCTCGGGTGCCAAGGA -o SRR22739469_trimmed.fastq.gz SRR22739469.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACGTCT -o SRR17652211_trimmed.fastq.gz SRR17652211.fastq.gz
# TODO: these files were not processed correctly as UGAGGUAGGAGGUUGUAUAGUU is not a adapter
cutadapt -j 16 -e 0.1 -a $ADAPTER -o SRR17652201_trimmed.fastq.gz SRR17652201.fastq.gz
cutadapt -j 16 -e 0.1 -a $ADAPTER -o SRR10240206_trimmed.fastq.gz SRR10240206.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGC -o SRR10240195_trimmed.fastq.gz SRR10240195.fastq.gz
cutadapt -j 16 -e 0.1 -a AGATCGGAAGAGCACACGTCTGA -o SRR8494806_trimmed.fastq.gz SRR8494806.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCCAAGGAAC -o SRR8494810_trimmed.fastq.gz SRR8494810.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCCAAGGAACT -o SRR6793389_trimmed.fastq.gz SRR6793389.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCCAAGGAACT -o SRR6793401_trimmed.fastq.gz SRR6793401.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCCAAGGAAC -o SRR6787013_trimmed.fastq.gz SRR6787013.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCGGGTGCCAAGGAAC -o SRR6787012_trimmed.fastq.gz SRR6787012.fastq.gz
cutadapt -j 16 -e 0.1 -a GGAATTCTCG -o SRR6787016_trimmed.fastq.gz SRR6787016.fastq.gz

