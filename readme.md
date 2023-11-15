# Mirdeepsquared 
Mirdeepsquared uses a deep learning model that predicts if novel miRNA sequences in the output of miRDeep2 are false positives or not. This greatly reduces the amount of manual work that is currently needed to filter out the false positives.

## Usage (with pip install)
virtualenv mirdeepsquared-env -p python3.9
source mirdeepsquared-env/bin/activate
pip install mirdeepsquared==0.1.1
predict path/to/your_result.csv path/to/your_output.mrd

The output are your true positives (i.e likely to actually be novel miRNA:s)

## Installing (from source)
Use python 3.9 as tensorflow requires it

```
virtualenv mirdeepsquared-env -p python3.9
source mirdeepsquared-env/bin/activate
pip install -r requirements.txt
python train-simple-density-map.py
```

### Installing on Uppmax
```
git clone https://github.com/jontejj/mirdeepsquared.git
cd mirdeepsquared
module load python3/3.9.5
virtualenv mirdeepsquared-env -p python3.9
source mirdeepsquared-env/bin/activate
pip install -r requirements.txt
python train-simple-density-map.py
```

Then you can use ```python predict.py your_result.csv your_output.mrd``` to get a list of the true positives

# How Mirdeepsquared was developed

https://www.ncbi.nlm.nih.gov/sra was used to select SRR datafiles. The accession list was then used to download the datafiles with fasterq-dump:

```
fasterq-dump -e 16 -t temp SRR_ID
```
```cutadapt``` was then used to trim adapters. The resulting files where then listed in a ```config.txt``` file like this:
```
SRR2496781.fastq hh1
SRR2496782.fastq hh2
SRR2496783.fastq hh3
SRR2496784.fastq hh4
```

A bowtie index (GRCh38 (from https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.39/)) for the human genome was built with https://bowtie-bio.sourceforge.net/manual.shtml:
```
bowtie-build GRCh38.p13.genome.fa GRCh38 --threads 8
```

Then mapper.pl (from miRDeep2) was used to create ```healthy_reads_collapsed.fa``` and ```healthy_reads_collapsed_vs_genome.arf``` from the reads:
```
mapper.pl config.txt -d -e -m -p GRCh38 -s healthy_reads_collapsed.fa -t healthy_reads_collapsed_vs_genome.arf -v -h
```

```precursor-sequences-h_sapiens.fas```, ```rhesus-monkey-mature-seq.fas``` and ```known-mature-sequences-h_sapiens.fas``` was downloaded from https://mirgenedb.org/. Then miRDeep2.pl was used to create the output.mrd and the result*.csv file:
```
miRDeep2.pl healthy_reads_collapsed.fa GRCh38.p13.genome.fa healthy_reads_collapsed_vs_genome.arf known-mature-sequences-h_sapiens.fas rhesus-monkey-mature-seq.fas precursor-sequences-h_sapiens.fas -t Human -b -5 2>report2.log
```

This process was done for one healthy tissue (SRR2496781-SRR2496784), where all the novel classifications were marked as false positives and then it was also done for data from the TCGA dataset (https://www.cancer.gov/ccg/research/genome-sequencing/tcga) (specifially TCGA-LUSC), where the mature classifications were assumed to be true positives. For the false positives, -b -5 was added to the miRDeep2.pl command to include even more false positives than miRDeep2.pl normally gives.

The resulting output.mrd and result.csv was then passed to ```extract_features.py``` in order to create the dataset (pickle files with Pandas dataframe in them):
True positives:
```
python extract_features.py TCGA-LUSC/result_19_01_2023_t_23_35_49.csv TCGA-LUSC/output.mrd resources/dataset/true_positives_TCGA_LUSC.pkl -m resources/known-mature-sequences-h_sapiens.fas -tp
```
False positives:
```
python extract_features.py false_positives/result_08_11_2023_t_19_35_00.csv false_positives/08_11_2023_t_19_35_00_output.mrd resources/dataset/false_positives_SRR2496781-84_bigger.pkl -m resources/known-mature-sequences-h_sapiens.fas -fp
```

As the resulting dataset files, ```true_positives_TCGA_LUSC.pkl``` and ```false_positives_SRR2496781-84_bigger.pkl```, were small, they were checked into git in resources/dataset/ in order to make future training easier for people who want to improve the model. On top of this, the output.mrd and the result.csv for the false positives were also checked in to git.

The different train*.py files contain different models with varying performance. The best one so far is ```train-simple-density-map.py```, it gave 100% accuracy on the test set.

To test how well the model generalizes a dataset was also created for Zebrafish (Danio rerio). Reference genome: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000002035.6/, accesision files: https://www.ncbi.nlm.nih.gov/Traces/study/?page=6&query_key=1&WebEnv=MCID_654de450ecadc040f52345a9&f=assay_type_s%3An%3Amirna-seq%3Ac&o=run_file_create_date_dt%3Ad%253Bacc_s%3Bacc_s%3Aa&s=SRR8305633,SRR6411465,SRR8305629,SRR8305619,SRR10358540,SRR6411467,SRR6411468,SRR6411466,SRR15498151,SRR15498149,SRR11974581,SRR11974578.