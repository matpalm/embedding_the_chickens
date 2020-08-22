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
    def __init__(self):
        self.run = []
        self.df_records = []

    def add(self, fname_dir, count):
        self.run.append(Crop(fname_dir, count))

    def flush(self):
        if len(self.run) >= 2:
            c0 = self.run.pop(0)
            for c1 in self.run:
                if c0.count <= c1.count:
                    self.df_records.append((base_dir, c0.fname, c1.fname))
                else:
                    self.df_records.append((base_dir, c1.fname, c0.fname))
                c0 = c1
        self.run = []


run = Run()
for cam in ['pi_a', 'pi_b', 'pi_c']:
    for dts in sorted(os.listdir(f"crops/{cam}")):
        base_dir = f"crops/{cam}/{dts}"
        for fname_dir in sorted(os.listdir(base_dir)):
            count = len(glob.glob(f"{base_dir}/{fname_dir}/*png"))
            if count >= 5:
                run.add(fname_dir, count)
            else:
                run.flush()

# final flush
run.flush()

# write out tsv
egs = run.df_records
random.shuffle(egs)
split_idx = int(len(egs) * 0.9)
train_records = run.df_records[:split_idx]
test_records = run.df_records[split_idx:]

pd.DataFrame(train_records, columns=['dir', 'frame_0', 'frame_1']).to_csv(
    "train.tsv", sep="\t", index=False)
pd.DataFrame(test_records, columns=['dir', 'frame_0', 'frame_1']).to_csv(
    "test.tsv", sep="\t", index=False)
