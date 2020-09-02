import argparse
import os
import glob
import pandas as pd
from collections import namedtuple
import sys
import random

# step through frames and collect cases where there are 5+ crops for two
# frames in a row. collect all these in a dataframe as a pair where the
# first in the pair has the least crops.

# e.g. for (fname, count) sequence of [(f0, 6), (f1, 7), (f2, 5), (f3, 4)]
# we'd emit [(f0, f1), (f2, f1)]

Crop = namedtuple('Crop', ['fname', 'count'])


class Run(object):
    def __init__(self, base_dir, df_records):
        self.run = []
        self.base_dir = base_dir
        self.df_records = df_records

    def add(self, fname_dir, count):
        self.run.append(Crop(fname_dir, count))

    def flush(self):
        if len(self.run) >= 2:
            c0 = self.run.pop(0)
            for c1 in self.run:
                f0 = c0.fname
                f1 = c1.fname
                # we want f0 entry represent lower count, so if this is
                # not the case swap f0 & f1
                if c0.count > c1.count:
                    f0, f1 = f1, f0
                rec = f"{self.base_dir}/{f0}", f"{self.base_dir}/{f1}"
                self.df_records.append(rec)
                c0 = c1
        self.run = []


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train-tsv', type=str, default='train.tsv')
parser.add_argument('--test-tsv', type=str, default='test.tsv')
opts = parser.parse_args()
print(opts, file=sys.stderr)

df_records = []
for cam in ['pi_a', 'pi_b', 'pi_c']:
    for dts in sorted(os.listdir(f"crops/{cam}")):
        base_dir = f"crops/{cam}/{dts}"
        run = Run(base_dir, df_records)
        for fname_dir in sorted(os.listdir(base_dir)):
            count = len(glob.glob(f"{base_dir}/{fname_dir}/*png"))
            if count >= 5:
                run.add(fname_dir, count)
            else:
                run.flush()
        run.flush()

# sanity check
for f0, f1 in df_records:
    if not os.path.exists(f"{f0}/crops.npy"):
        print("no crops for ", base_dir, f0)
    if not os.path.exists(f"{f1}/crops.npy"):
        print("no crops for ", base_dir, f1)

# write out tsv
random.seed(42)
random.shuffle(df_records)
split_idx = int(len(df_records) * 0.9)
train_records = run.df_records[:split_idx]
test_records = run.df_records[split_idx:]

pd.DataFrame(train_records, columns=['frame_0', 'frame_1']).to_csv(
    opts.train_tsv, sep="\t", index=False)
pd.DataFrame(test_records, columns=['frame_0', 'frame_1']).to_csv(
    opts.test_tsv, sep="\t", index=False)
