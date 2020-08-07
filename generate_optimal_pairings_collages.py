from PIL import Image
import numpy as np
from random_embeddings.ensemble_net import EnsembleNet
from random_embeddings.optimal_pairing import calculate_optimal_pairing
import os
import glob
import pandas as pd

# given dataset of adjacent frames (from successive_frames_to_training_data)
# generate embeddings from random ensemble net, calculate pair wise sims,
# derive optimal pairing and generate collage under "collages/"


def load_crops_as_floats(fname):
    return np.load(fname).astype(np.float32) / 255.0


def pil_img_from_array(array):
    return Image.fromarray((array * 255.0).astype(np.uint8))


def collage(examples):
    if len(examples.shape) != 5:
        raise Exception("Expected grid of examples; (R, C, H, W, 3)")
    if examples.shape[2] != examples.shape[3]:
        raise Exception("Expected examples to be square; i.e. H==W")
    HW = examples.shape[2]
    HWB = HW + 2  # height / width with 2 pixel buffer for collage
    num_rows, num_cols = examples.shape[:2]
    collage = Image.new('RGB', (num_cols*HWB, num_rows*HWB))
    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            img = pil_img_from_array(examples[r_idx, c_idx])
            collage.paste(img, (c_idx*HWB, r_idx*HWB))
    return collage.resize((2*num_cols*HWB, 2*num_rows*HWB))


def score_optimal_pairing(optimal_pairing, sims):
    return sum([sims[i, j] for i, j in optimal_pairing.items()])


def optimal_pairing_collage(optimal_pairing, crops0, crops1):
    # keep record all all indexs from both sets of crops
    # so at end we can fill in ones that weren't in optimal_pairing
    idxs_from_crops0 = list(range(len(crops0)))
    idxs_from_crops1 = list(range(len(crops1)))

    # make collage placeholder, 2 rows with enough columns to
    # fit longer of crops0 or crops1
    num_cols = max([len(crops0), len(crops1)])
    single_img_shape = crops0[0].shape
    collage_imgs = np.zeros((2, num_cols, *single_img_shape))

    # iterate through optimal pairing filling in columns from left to right
    for col, (c0, c1) in enumerate(optimal_pairing.items()):
        collage_imgs[0, col] = crops0[c0]
        collage_imgs[1, col] = crops1[c1]
        idxs_from_crops0.remove(c0)
        idxs_from_crops1.remove(c1)

    # pad remainining columns with left over; will come from either crops0
    # or crops1
    for i, c0 in enumerate(idxs_from_crops0):
        collage_imgs[0, col+1+i] = crops0[c0]
    for i, c1 in enumerate(idxs_from_crops1):
        collage_imgs[1, col+1+i] = crops1[c1]

    return collage(collage_imgs)


if __name__ == '__main__':
    ensemble_net = EnsembleNet(num_models=10)

    df = pd.read_csv("dts_f0_f1.tsv", sep="\t", dtype=object)

    n = 0
    for _, row in df.iterrows():

        base_dir, f0, f1 = row['dir'], row['frame_0'], row['frame_1']
        print(base_dir, f0, f1)

        crops_t0 = load_crops_as_floats(f"{base_dir}/{f0}/crops.npy")
        embeddings_t0 = ensemble_net.embed(crops_t0)

        crops_t1 = load_crops_as_floats(f"{base_dir}/{f1}/crops.npy")
        embeddings_t1 = ensemble_net.embed(crops_t1)

        sims = np.einsum('mae,mbe->ab', embeddings_t0, embeddings_t1)
        print("sims", sims.shape)
        print(np.around(sims, decimals=2))

        optimal_pairing = calculate_optimal_pairing(sims)
        print("score_optimal_pairing",
              score_optimal_pairing(optimal_pairing, sims))

        labels = np.zeros_like(sims)
        for i, j in optimal_pairing.items():
            labels[i, j] = 1.0
        print(labels)

        collage_fname = f"collages/optimal_pairing.{f0}_{f1}.png"
        optimal_pairing_collage(
            optimal_pairing, crops_t0, crops_t1).save(collage_fname)

        n += 1
        if n == 3:
            exit()
