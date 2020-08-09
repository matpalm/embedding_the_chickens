from PIL import Image
import numpy as np


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
