import time
from embed_net import optimal_pairing
from embed_net import ensemble_net
from detections import img_utils
from jax import jit, grad
from jax.experimental import optimizers
import pandas as pd
import random as orandom
import wandb
import sys


def parse_frame_pairs():
    # read all crop frame pairs from training data .tsv
    df = pd.read_csv("dts_f0_f1.tsv", sep="\t", dtype=object)
    frame_pairs = []
    for _index, row in df.iterrows():
        frame_pairs.append((f"{row['dir']}/{row['frame_0']}",
                            f"{row['dir']}/{row['frame_1']}"))
    # <hack>
    orandom.seed(0)
    orandom.shuffle(frame_pairs)
    frame_pairs = frame_pairs[:100]
    # </hack>
    return frame_pairs


def train(opts):
    # init w & b
    wandb.init(project='embedding_the_chickens',
               group=opts.group, name=opts.run)
    wandb.config.num_models = opts.num_models
    wandb.config.learning_rate = opts.learning_rate

    # init model
    params = ensemble_net.initial_params(num_models=opts.num_models,
                                         seed=opts.seed)

    # init optimiser
    opt_fns = optimizers.adam(step_size=opts.learning_rate)
    opt_init_fun, opt_update_fun, opt_get_params = opt_fns
    opt_state = opt_init_fun(params)

    # create a jitted step function
    @jit
    def step(i, opt_state, crops_t0, crops_t1, labels):
        params = opt_get_params(opt_state)
        gradients = grad(ensemble_net.loss)(params, crops_t0, crops_t1, labels)
        new_opt_state = opt_update_fun(i, gradients, opt_state)
        return new_opt_state

    # run training loop
    frame_pairs = parse_frame_pairs()
    i = 0
    for e in range(opts.epochs):
        orandom.seed(e)
        orandom.shuffle(frame_pairs)
        for f0, f1 in frame_pairs:
            i += 1
            try:
                # load
                crops_t0 = img_utils.load_crops_as_floats(f"{f0}/crops.npy")
                crops_t1 = img_utils.load_crops_as_floats(f"{f1}/crops.npy")
                # forward pass
                sims = ensemble_net.calc_sims(params, crops_t0, crops_t1)
                # derive labels from optimal pairing
                pairing = optimal_pairing.calculate(sims)
                labels = optimal_pairing.to_one_hot_labels(sims, pairing)
                # take update step
                opt_state = step(i, opt_state, crops_t0, crops_t1, labels)
                params = opt_get_params(opt_state)
                # print loss for debugging
                loss = ensemble_net.loss(params, crops_t0, crops_t1, labels)
                # line of debug
                print(e, i, f0, f1, loss)
                wandb.log({'loss': float(loss)})
            except Exception as e:
                print("exception", str(e), e, i, f0, f1, file=sys.stderr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--group', type=str,
                        help='w&b init group', default=None)
    parser.add_argument('--run', type=str,
                        help='w&b init run', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-models', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    opts = parser.parse_args()
    train(opts)
