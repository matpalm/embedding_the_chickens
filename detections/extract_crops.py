import argparse
from db import img_db
import file_util
from PIL import Image
import numpy as np
import detections.util as u
import sys
from collections import Counter
from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--entity-allow-list', type=str,
                    help='comma seperated list of entities to ignore')
parser.add_argument('--entity-deny-list', type=str,
                    help='comma seperated list of entities to ignore')
parser.add_argument('--manifest', type=str, default=None)
parser.add_argument('--min-score', type=float, default=0.0)
opts = parser.parse_args()
print("opts %s" % opts, file=sys.stderr)

allow_deny_filter = u.AllowDenyFilter(opts.entity_allow_list,
                                      opts.entity_deny_list)
db = img_db.ImgDB()
crop_entity_freq = Counter()
ignore_freq = Counter()

fnames = [s.strip() for s in open(opts.manifest).readlines()]
for fname in tqdm(fnames):

    # do first pass filter of entity allow/deny and score
    filtered_detections = []
    for d in db.detections_for_img(fname):
        if not allow_deny_filter.allow(d.entity):
            ignore_freq[d.entity] += 1
        elif d.score < opts.min_score:
            ignore_freq["BELOW_MIN_SCORE"] += 1
        else:
            filtered_detections.append(d)
    if len(filtered_detections) == 0:
        ignore_freq['ALL_FILTERED'] += 1
        continue

    # do second pass filter based on non_max_suppression
    filtered_detections = u.non_max_suppression(filtered_detections,
                                                overlap_thresh=0.6)

    cam, ymd, hms = file_util.split_fname(fname)
    crop_dir = f"crops/{cam}/{ymd}/{hms}"
    file_util.ensure_dir_exists(crop_dir)
    pil_img = Image.open(fname)
    crop_np_arrays = []
    crop_detection_ids = []
    for d in filtered_detections:

        # make bb crop square (with buffer) resized to (128, 128)
        x0, y0, x1, y1 = u.square_bb(d.x0-5, d.y0-5, d.x1+5, d.y1+5)
        crop = pil_img.crop((x0, y0, x1, y1))
        crop = crop.resize((128, 128), Image.LANCZOS)

        # save a PNG for debugging
        crop_fname = "%s/%08d.%s.%0.2f.png" % (crop_dir,
                                               d.id, d.entity, d.score)
        crop.save(crop_fname)  # , quality=100, optimize=True)

        # collect crop np array and detection id for later storing stack
        crop_np_arrays.append(np.array(crop))
        crop_detection_ids.append(d.id)

        # collect some entity stats
        crop_entity_freq[d.entity] += 1

    np.save(f"{crop_dir}/crops.npy", np.stack(crop_np_arrays))
    np.save(f"{crop_dir}/crop_detection_ids.npy", crop_detection_ids)

print()
print("crop_entity_freq", crop_entity_freq)
print("ignore_freq", ignore_freq)
