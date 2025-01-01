import argparse
import sys
import pandas as pd
from mirdeepsquared.common import prepare_data

import ViennaRNA

def inspect(file, entry):
    df = pd.read_pickle(file)
    df = prepare_data(df)
    columns = df.columns
    single_item = df[(df['location'] == entry)]
    if len(single_item) == 1:
        for column in columns:
            print(column + ":" + str(single_item[column].values[0]))
    else:
        print("No entry like " + str(entry) + " found")


def parse_args(args):
    parser = argparse.ArgumentParser(prog='MirDeepSquared-inspect', description='Print data for one sample')

    parser.add_argument('dataset')  # positional argument
    parser.add_argument('-e', '--entry', help="The entry to inspect, without this the entries are listed instead", default=argparse.SUPPRESS)  # optional argument
    return parser.parse_args(args)


if __name__ == '__main__':
    seq = "AGACGACAAGGUUGAAUCGCACCCACAGUCUAUGAGUCGGUG"
    fc = ViennaRNA.fold_compound(seq)
    (ss, mfe) = fc.mfe()  # maximum -15 kcal/mol free energy (Brameier & Wiuf, 2007)
    print(f"{seq}\n{ss} ({mfe:6.2f})")

    args = parse_args(sys.argv[1:])
    # args = parse_args(['resources/dataset/other_species/true_positives/zebrafish/zebrafish.mature.2nd.run.pkl', 'AAA'])
    if hasattr(args, 'entry'):
        inspect(args.dataset, args.entry)
    else:
        df = pd.read_pickle(args.dataset)
        for location in df['location'].values:
            print(location)

