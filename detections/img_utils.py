from PIL import Image
import numpy as np
import jax.numpy as jnp
from functools import lru_cache
import pandas as pd

# NOTE: no bound on cache; i.e. assume entire dataset fits in GPU mem
@lru_cache(None)
def load_crops_as_floats(fname):
    np_array = np.load(fname).astype(np.float32) / 255.0
    return jnp.array(np_array)


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


def parse_frame_pairs(manifest):
    df = pd.read_csv(manifest, sep="\t", dtype=object)
    frame_pairs = []
    for _index, row in df.iterrows():
        frame_pairs.append((row['frame_0'], row['frame_1']))
    return frame_pairs
