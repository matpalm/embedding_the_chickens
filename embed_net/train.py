import time
import numpy as np
from embed_net import optimal_pairing
from embed_net import ensemble_net
from detections import img_utils
from jax import jit, grad
from jax.experimental import optimizers
import pandas as pd
import random as orandom
import wandb
import sys
import traceback
import datetime

#from jax.config import config
#config.update("jax_debug_nans", True)


def parse_frame_pairs(manifest):
    # read all crop frame pairs from training data .tsv
    df = pd.read_csv(manifest, sep="\t", dtype=object)
    frame_pairs = []
    for _index, row in df.iterrows():
        frame_pairs.append((f"{row['dir']}/{row['frame_0']}",
                            f"{row['dir']}/{row['frame_1']}"))
    return frame_pairs


def train(opts):

    train_frame_pairs = parse_frame_pairs(opts.train_tsv)
    test_frame_pairs = parse_frame_pairs(opts.test_tsv)

    # init w & b
    wandb_enabled = opts.group is not None
    if wandb_enabled:
        if opts.run is None:
            run = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            run = opts.run
        wandb.init(project='embedding_the_chickens',
                   group=opts.group, name=run)
        wandb.config.num_models = opts.num_models
        wandb.config.learning_rate = opts.learning_rate
        wandb.config.epochs = opts.epochs
    else:
        print("not using wandb", file=sys.stderr)

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

    def labels_for_crops(params, crops_t0, crops_t1):
        # forward pass
        sims = ensemble_net.calc_sims(params, crops_t0, crops_t1)
        # derive labels from optimal pairing
        pairing = optimal_pairing.calculate(sims)
        labels = optimal_pairing.to_one_hot_labels(sims, pairing)
        return labels

    # run training loop
    i = 0
    for e in range(opts.epochs):
        # shuffle training examples
        orandom.seed(e)
        orandom.shuffle(train_frame_pairs)

        # make pass through training examples
        train_losses = []
        for f0, f1 in train_frame_pairs:
            i += 1
            try:
                # load
                crops_t0 = img_utils.load_crops_as_floats(f"{f0}/crops.npy")
                crops_t1 = img_utils.load_crops_as_floats(f"{f1}/crops.npy")
                # calc labels for crops
                labels = labels_for_crops(params, crops_t0, crops_t1)
                # take update step
                opt_state = step(i, opt_state, crops_t0, crops_t1, labels)
                params = opt_get_params(opt_state)
                # collect loss for debugging
                loss = ensemble_net.loss(params, crops_t0, crops_t1, labels)
                train_losses.append(float(loss))
                # print line of debug
                print(e, i, f0, f1, loss)
            except Exception:
                print("train exception", e, i, f0, f1, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # eval mean test loss
        test_losses = []
        params = opt_get_params(opt_state)
        for f0, f1 in test_frame_pairs:
            try:
                # load
                crops_t0 = img_utils.load_crops_as_floats(f"{f0}/crops.npy")
                crops_t1 = img_utils.load_crops_as_floats(f"{f1}/crops.npy")
                # calc labels for crops
                labels = labels_for_crops(params, crops_t0, crops_t1)
                # collect loss
                loss = ensemble_net.loss(params, crops_t0, crops_t1, labels)
                test_losses.append(loss)
            except Exception:
                print("test exception", e, i, f0, f1, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # log stats
        mean_train_loss = np.mean(train_losses)
        mean_test_loss = np.mean(test_losses)
        print("mean train loss", mean_train_loss)
        print("mean test loss", mean_test_loss)
        nan_loss = np.isnan(mean_train_loss) or np.isnan(mean_test_loss)
        if wandb_enabled and not nan_loss:
            wandb.log({'train_loss': np.mean(train_losses)})
            wandb.log({'test_loss': np.mean(test_losses)})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--group', type=str,
                        help='w&b init group', default=None)
    parser.add_argument('--run', type=str,
                        help='w&b init run', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-tsv', type=str,
                        default='manifests/20200811/train/sample_20.tsv')
    parser.add_argument('--test-tsv', type=str,
                        default='manifests/20200811/train/sample_20.tsv')
    parser.add_argument('--num-models', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=3)
    opts = parser.parse_args()
    print(opts, file=sys.stderr)
    train(opts)
