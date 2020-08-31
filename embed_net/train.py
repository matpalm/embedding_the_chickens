import time
import numpy as np
from embed_net import optimal_pairing
from embed_net import ensemble_net
from detections import img_utils
import random as orandom
import wandb
import sys
import traceback
import datetime
from tqdm import tqdm
from itertools import count
import objax

# uncomment for nan check
#from jax.config import config
#config.update("jax_debug_nans", True)

# XLA_PYTHON_CLIENT_MEM_FRACTION=.XX


def train(opts):
    if opts.ortho_init in ['True', 'False']:
        # TODO: move this to argparses problem
        opts.ortho_init = opts.ortho_init == 'True'
    if opts.ortho_init not in [True, False]:
        raise Exception("unknown --ortho-init value [%s]" % opts.ortho_init)
    ortho_init = opts.ortho_init == 'True'

    train_frame_pairs = img_utils.parse_frame_pairs(opts.train_tsv)
    test_frame_pairs = img_utils.parse_frame_pairs(opts.test_tsv)

    # init w & b
    wandb_enabled = opts.group is not None
    if wandb_enabled:
        if opts.run is None:
            run = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            run = opts.run
        wandb.init(project='embedding_the_chickens',
                   group=opts.group, name=run,
                   reinit=True)
        wandb.config.num_models = opts.num_models
        wandb.config.dense_kernel_size = opts.dense_kernel_size
        wandb.config.embedding_dim = opts.embedding_dim
        wandb.config.learning_rate = opts.learning_rate
        wandb.config.ortho_init = opts.ortho_init
        wandb.config.logit_temp = opts.logit_temp

    else:
        print("not using wandb", file=sys.stderr)

    # init model
    embed_net = ensemble_net.EnsembleNet(num_models=opts.num_models,
                                         dense_kernel_size=opts.dense_kernel_size,
                                         embedding_dim=opts.embedding_dim,
                                         seed=opts.seed,
                                         orthogonal_init=ortho_init)

    # make some jitted version of embed_net methods
    # TODO: objax decorator to do this?
    j_embed_net_calc_sims = objax.Jit(embed_net.calc_sims, embed_net.vars())
    j_embed_net_loss = objax.Jit(embed_net.loss, embed_net.vars())

    # run pair of crops through model and calculate optimal pairing
    def labels_for_crops(crops_t0, crops_t1):
        sims = j_embed_net_calc_sims(crops_t0, crops_t1)
        pairing = optimal_pairing.calculate(sims)
        labels = optimal_pairing.to_one_hot_labels(sims, pairing)
        return labels

    # init optimiser
    gradient_loss = objax.GradValues(embed_net.loss, embed_net.vars())
    optimiser = objax.optimizer.Adam(embed_net.vars())
    lr = 1e-3

    # create a jitted training step
    def train_step(crops_t0, crops_t1, labels):
        grads, loss = gradient_loss(crops_t0, crops_t1, labels)
        optimiser(lr, grads)
        return loss

    train_step = objax.Jit(train_step, gradient_loss.vars() + optimiser.vars())

    # run training loop
    for e in range(opts.epochs):
        # shuffle training examples
        orandom.seed(e)
        orandom.shuffle(train_frame_pairs)

        # make pass through training examples
        train_losses = []
        for f0, f1 in tqdm(train_frame_pairs):
            try:
                # load
                crops_t0 = img_utils.load_crops_as_floats(f"{f0}/crops.npy")
                crops_t1 = img_utils.load_crops_as_floats(f"{f1}/crops.npy")
                # calc labels for crops
                labels = labels_for_crops(crops_t0, crops_t1)
                # take update step
                loss = train_step(crops_t0, crops_t1, labels)
                train_losses.append(loss)
            except Exception:
                print("train exception", e, f0, f1, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # eval mean test loss
        test_losses = []
        for f0, f1 in test_frame_pairs:
            try:
                # load
                crops_t0 = img_utils.load_crops_as_floats(f"{f0}/crops.npy")
                crops_t1 = img_utils.load_crops_as_floats(f"{f1}/crops.npy")
                # calc labels for crops
                labels = labels_for_crops(crops_t0, crops_t1)
                # collect loss
                loss = j_embed_net_loss(crops_t0, crops_t1, labels)
                test_losses.append(loss)
            except Exception:
                print("test exception", e, f0, f1, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # log stats
        mean_train_loss = np.mean(train_losses)
        mean_test_loss = np.mean(test_losses)
        print(e, mean_train_loss, mean_test_loss)

        # TODO: or only if train_loss is Nan?
        nan_loss = np.isnan(mean_train_loss) or np.isnan(mean_test_loss)
        if wandb_enabled and not nan_loss:
            wandb.log({'train_loss': np.mean(train_losses)})
            wandb.log({'test_loss': np.mean(test_losses)})

    # close out wandb run
    wandb.join()

    # note: use None value to indicate run failed
    if nan_loss:
        return None
    else:
        return mean_test_loss


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
    parser.add_argument('--dense-kernel-size', type=int, default=32)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--ortho-init', type=str, default='True')
    parser.add_argument('--logit-temp', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=3)
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    train(opts)
