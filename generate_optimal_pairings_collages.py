from PIL import Image
import numpy as np
from embed_net import ensemble_net
from embed_net import optimal_pairing
import os
import glob
import pandas as pd

# given dataset of adjacent frames (from successive_frames_to_training_data)
# generate embeddings from random ensemble net, calculate pair wise sims,
# derive optimal pairing and generate collage under "collages/"


def load_crops_as_floats(fname):
    return np.load(fname).astype(np.float32) / 255.0


if __name__ == '__main__':
    params = ensemble_net.initial_params(num_models=10)

    df = pd.read_csv("dts_f0_f1.tsv", sep="\t", dtype=object)

    n = 0
    for _, row in df.iterrows():

        base_dir, f0, f1 = row['dir'], row['frame_0'], row['frame_1']
        print(base_dir, f0, f1)

        crops_t0 = load_crops_as_floats(f"{base_dir}/{f0}/crops.npy")
        embeddings_t0 = ensemble_net.embed(params, crops_t0)

        crops_t1 = load_crops_as_floats(f"{base_dir}/{f1}/crops.npy")
        embeddings_t1 = ensemble_net.embed(params, crops_t1)

        sims = np.einsum('mae,mbe->ab', embeddings_t0, embeddings_t1)
        print("sims", sims.shape)
        print(np.around(sims, decimals=2))

        pairing = optimal_pairing.calculate(sims)
        print("optimal_pairing.score", optimal_pairing.score(pairing, sims))

        labels = np.zeros_like(sims)
        for i, j in pairing.items():
            labels[i, j] = 1.0
        print(labels)

        collage_fname = f"collages/optimal_pairing.{f0}_{f1}.png"
        optimal_pairing.collage(
            pairing, crops_t0, crops_t1).save(collage_fname)

        n += 1
        if n == 3:
            exit()
