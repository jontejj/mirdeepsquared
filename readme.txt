#Use python 3.9 as tensorflow requires it
virtualenv miRNA -p python3.9

pip install -r requirements.txt

#Uppmax
git clone https://github.com/jontejj/mirdeepsquared.git
cd mirdeepsquared
module load python3/3.9.5
virtualenv miRNA -p python3.9
source miRNA/bin/activate
pip install -r requirements.txt
