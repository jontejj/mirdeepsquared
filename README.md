<meta name="DC.Creator" content="Jonatan Jönsson">
<meta name="DC.Title" content="Mirdeepsquared: a deep learning model that predicts if novel miRNA sequences are false positives or not">

# Mirdeepsquared
Mirdeepsquared uses a deep learning model that predicts if novel miRNA sequences in the output of [miRDeep2](https://github.com/rajewsky-lab/mirdeep2) are false positives or not. This greatly reduces the amount of manual work that is currently needed to filter out the false positives.

## Simple usage (with pip install)
```
virtualenv mirdeepsquared-env -p python3.9
source mirdeepsquared-env/bin/activate
pip install mirdeepsquared
mirdeepsquared path/to/your_result.csv path/to/your_output.mrd
```

The output are your true positives, i.e highly likely to actually be novel miRNA:s.

Another way to use mirdeepsquared is to create a filtered result.html file:
```
mirdeepsquared path/to/your_result.csv path/to/your_output.mrd > tp.txt
mod-html path/to/your_result.html path/to/filtered_results.html -- tp.txt
```

## Prerequiste for installing on Apple Silicone (M-processors)

The first step is to make sure that the python 3.9 version was built for the arm architecture.
```
python3.9 -c "import platform; print(platform.machine())"
```

If this prints x86_64, python3.9 points to python with the wrong architecture. In order to fix this you can first change the default shell to Z shell (/bin/zsh):
```
chsh -s /bin/zsh
```

Then start a new terminal and make sure that zsh is used and not bash for example. Then install homebrew for the right architecture:
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
Also make sure to run the commands at the end of the output to add Homebrew to your PATH.

Then run:
```
brew install python@3.9
```
Verify that arm64 / arm64e is output when running:
```
/opt/homebrew/bin/python3.9 -c "import platform; print(platform.machine())"
```

## Installing (from source)
In order not to clutter your global python install, it's better to use a virtual environment:
Use python 3.9 as tensorflow requires it.
If homebrew was used to install python 3.9 make sure that:
/opt/homebrew/bin/python3.9 is the python version used.

```
virtualenv mirdeepsquared-env -p python3.9
source mirdeepsquared-env/bin/activate
pip install -r requirements.txt
python3 -m pip install -e .
./prepare_default_dataset.sh
python mirdeepsquared/train.py resources/dataset/split/train -o models/
python mirdeepsquared/predict_cmd.py path/to/your_result.csv path/to/your/output.mrd -m models/
```

### Installing on Uppmax
```
git clone https://github.com/jontejj/mirdeepsquared.git
cd mirdeepsquared
module load python3/3.9.5
virtualenv mirdeepsquared-env -p python3.9
source mirdeepsquared-env/bin/activate
pip install -r requirements.txt
./prepare_default_dataset.sh
python mirdeepsquared/train.py resources/dataset/split/train -o models/
```

Then you can use ```python mirdeepsquared/predict_cmd.py your_result.csv your_output.mrd -m models/``` to get a list of the true positives

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
python mirdeepsquared/extract_features.py TCGA-LUSC/result_19_01_2023_t_23_35_49.csv TCGA-LUSC/output.mrd resources/dataset/true_positives/true_positives_TCGA_LUSC_all.pkl -tp --section known
python mirdeepsquared/mirgene_db_filter.py resources/dataset/true_positives/true_positives_TCGA_LUSC_all.pkl resources/known-mature-sequences-h_sapiens.fas resources/dataset/true_positives_TCGA_LUSC_only_in_mirgene_db.pkl"
```
False positives:
```
python extract_features.py false_positives/result_08_11_2023_t_19_35_00.csv false_positives/08_11_2023_t_19_35_00_output.mrd resources/dataset/false_positives_SRR2496781-84_bigger.pkl -fp --section novel
```

As the resulting dataset files, ```true_positives_TCGA_LUSC.pkl``` and ```false_positives_SRR2496781-84_bigger.pkl```, were small, they were checked into git in resources/dataset/ in order to make future training easier for people who want to improve the model. On top of this, the output.mrd and the result.csv for the false positives were also checked in to git.

The data was then split into a holdout and a train dataset with ```split_dataset.py```:
```
python split_dataset.py resources/dataset/ resources/dataset/split
```

This created ```resources/dataset/split/holdout/holdout.pkl``` and ```resources/dataset/split/train/train.pkl```.

The different train*.py files contain different models with varying performance. The best one so far is ```train-simple-density-map.py```, initially it got 100% accuracy on the test set. It later turned out that true positives not in mirgene db were not filtered away correctly ([#1][i1]), this lead to a less imbalanced dataset. The imbalance was later corrected by incorporating samples from TCGA_BRCA instead. With this correction, the accuracy was 98% instead.

To avoid overfitting to the training data the complexity of the model ```train-simple-density-map.py``` was reduced by lowering the units on the Dense layer (10000 -> 1000). The false negative and false positive samples in the confusion matrix (listed by ```predictor.py```) was then manually inspected. It turned out that some samples actually had the wrong label as they either were not really false positives or not really true positives. These labels were corrected with:
```
python mirdeepsquared/correct_invalid_labels.py
```

To test how well the model generalizes, a dataset was also created for Zebrafish ([Danio rerio][genome1]) and House mouse ([Mus musculus][genome2]). For these datafiles, the accuracy was first 88%. Then the mouse datafiles were included in the training data and the Zebrafish in the holdout dataset. The accuracy then improved to 90%.

By generating ```false_positives_with_empty_read_density_maps.pkl``` with ```generate_data.py``` the accuracy increased to 94.4% (Big cross-validated model (```trainer.py```), with best-hyperparameters.yaml, trained on: true_positives_TCGA_LUSC_only_in_mirgene_db.pkl + true_positives_TCGA_BRCA.pkl + mouse.mature.pkl + false_positives_with_empty_read_density_maps.pkl + false_positives_SRR2496781-84_bigger.pkl and evaluated on holdout + zebrafish.mature.2nd.run.pkl.

By combining ```density_map_model.py```, ```motifs_bayes_model.py```, ```numerical_model.py```, ```structure_model.py``` and the big model trained (in ```train.py```) with ```mirdeepsquared/best-hyperparameters.yaml``` into an ensemble model, the accuracy on the holdout data increased to 96%.

[release.sh](release.sh) was then used to release a mirdeepsquared package to https://pypi.org/.

The most problematic samples were then collected into pdf:s by ```predictor.py``` for analysis by experts used to filter out false positives from miRdeep2.

After analyzing the most problematic samples, some of the true positives were deemed not to actually be in mirgene db. It turned out that some of the known sequences matched a mature sequence in mirgene db but when the whole precursor was compared, there were differences. One possible explanation is that these could be pseudo genes that evolved from the functioning miRNA.

After making ```mirgene_db_filter.py``` more stringent, i.e instead of ```true_positives_TCGA_LUSC_only_in_mirgene_db.pkl```, ```true_positives_TCGA_LUSC_only_precursors_in_mirgene_db``` were generated and trained on instead. The accuracy, together with a new best-hyperparameters.yaml (based on the result of a half-finished grid search on uppmax), then increased to 99.3%.

*TODO*: analyze the most problematic samples in the subset created with ```resources/dataset/hard/only_matching_mature_in_mirgene_db/generate-tricky.sh```, and incorporate the tricky cases into the training data once it's been correctly labeled.

[i1]: https://github.com/jontejj/mirdeepsquared/issues/1

[genome1] : https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000002035.6/

[genome2]: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/

[acc1] : https://www.ncbi.nlm.nih.gov/Traces/study/?acc=SRR8305633%2CSRR6411465%2CSRR8305629%2CSRR8305619%2CSRR10358540%2CSRR6411467%2CSRR6411468%2CSRR6411466%2CSRR15498151%2CSRR15498149%2CSRR11974581%2CSRR11974578

[acc2] : https://www.ncbi.nlm.nih.gov/Traces/study/?acc=SRR25511174%2CSRR25338387%2CSRR25205118%2CSRR22739462%2CSRR24949848%2CSRR17652208%2CSRR10240201%2CSRR8494799%2CSRR6793409%2CSRR6787032%2CSRR1551219%2CSRR1551218%2CSRR1551220%2CSRR25205161%2CSRR25205080%2CSRR25338389%2CSRR25338388%2CSRR25511175%2CSRR25511182%2CSRR24949847%2CSRR24949826%2CSRR22739477%2CSRR22739469%2CSRR17652211%2CSRR17652201%2CSRR10240206%2CSRR10240195%2CSRR8494806%2CSRR8494810%2CSRR6793389%2CSRR6793401%2CSRR6787013%2CSRR6787012%2CSRR6787016&o=acc_s%3Aa