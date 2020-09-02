import argparse
from detections import img_utils
from embed_net import optimal_pairing
from embed_net import ensemble_net
from jax import jit
import pandas as pd
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--manifest-in', type=str,
                    default='manifests/20200811/train/sample_1.tsv')
parser.add_argument('--manifest-out', type=str,
                    default='foo.tsv')
parser.add_argument('--num-models', type=int, default=10)
parser.add_argument('--dense-kernel-size', type=int, default=32)
parser.add_argument('--embedding-dim', type=int, default=32)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ortho-init', type=str, default='True')
opts = parser.parse_args()
print(opts, file=sys.stderr)

# init model
embed_net = ensemble_net.EnsembleNet(num_models=opts.num_models,
                                     dense_kernel_size=opts.dense_kernel_size,
                                     embedding_dim=opts.embedding_dim,
                                     seed=opts.seed,
                                     orthogonal_init=(opts.ortho_init == 'True'))

# jit calc sims (we won't be updating it at all)
calc_sims = jit(embed_net.calc_sims)

# calculate optimal pairing for all entries in manifest_in
df_records = []
for f0, f1 in tqdm(img_utils.parse_frame_pairs(opts.manifest_in)):
    # load pairs
    crops_t0 = img_utils.load_crops_as_floats(f"{f0}/crops.npy")
    crops_t1 = img_utils.load_crops_as_floats(f"{f1}/crops.npy")
    # calc labels for crops
    sims = calc_sims(crops_t0, crops_t1)
    pairing = optimal_pairing.calculate(sims)
    # collect record for manifest_out
    df_records.append((f0, f1, pairing))

# write manifest_out
pd.DataFrame(df_records, columns=['frame_0', 'frame_1', 'pairing']).to_csv(
    opts.manifest_out, sep="\t", index=False)
