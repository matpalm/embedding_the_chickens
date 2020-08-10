#!/usr/bin/env python3
import os
import pandas as pd

# TODO: derive from first and check
CAMS = ['pi_a', 'pi_b', 'pi_c']

distinct_dts = set()
for cam in CAMS:
    distinct_dts.update(os.listdir(f"data/{cam}"))

rows = []
for dts in sorted(distinct_dts):
    row = [dts]
    for cam in CAMS:
        try:
            row.append(len(os.listdir(f"data/{cam}/{dts}")))
        except FileNotFoundError:
            row.append(0)
    rows.append(row)

print(pd.DataFrame(rows, columns=['dts'] + CAMS))
