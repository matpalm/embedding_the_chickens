
import argparse
from embed_net import train
from ax.service.ax_client import AxClient
import time
import sys
import file_util as u
import random

# TODO: fix the run layout; tb/RUN makes sense because of tensorboard
#       but logs/RUN.tsv should run RUN/log.tsv

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--group', type=str, default=None,
                    help='wandb group. if none, no logging')
# parser.add_argument('--prior-run-logs', type=str, default=None,
#                     help='a comma seperated prior runs to prime ax client with')
cmd_line_opts = parser.parse_args()
print(cmd_line_opts, file=sys.stderr)

ax = AxClient()
ax.create_experiment(
    name="embed_net_tuning",
    parameters=[
        # {
        #     "name": "num_models",
        #     "type": "range",
        #     "bounds": [5, 20],
        # },
        {
            "name": "dense_kernel_size",
            "type": "range",
            "bounds": [8, 64],
        },
        {
            "name": "embedding_dim",
            "type": "range",
            "bounds": [8, 64],
        },
        {
            "name": "learning_rate",
            "type": "range",
            "bounds": [1e-7, 1e-2],
            "log_scale": True,
        },
        # {
        #     "name": "ortho_init",
        #     "type": "choice",
        #     "value_type": "bool",
        #     "values": [True, False],
        # },
        {
            "name": "logit_temp",
            "type": "range",
            "bounds": [0.01, 10.0],
            "log_scale": True,
        },
    ],
    objective_name="final_loss",
    minimize=True,
)

# if cmd_line_opts.prior_run_logs is not None:
#     for log_tsv in cmd_line_opts.prior_run_logs.split(","):
#         u.prime_ax_client_with_prior_run(ax, log_tsv)

u.ensure_dir_exists("logs/%s" % cmd_line_opts.group)
log = open("logs/%s/ax_trials.tsv" % cmd_line_opts.group, "w")
print("trial_index\tparameters\truntime\ttest_score", file=log)


while True:
    parameters, trial_index = ax.get_next_trial()
    log_record = [trial_index, parameters]
    print("starting", log_record)

    class Opts(object):
        pass
    opts = Opts()

    # fixed opts
    opts.group = cmd_line_opts.group
    opts.run = None
    opts.seed = random.randint(0, 1e9)
    opts.train_tsv = 'manifests/20200811/train/all.tsv'
    opts.test_tsv = 'manifests/20200811/test.tsv'
    opts.num_models = 20  # parameters['num_models']
    opts.dense_kernel_size = parameters['dense_kernel_size']
    opts.embedding_dim = parameters['embedding_dim']
    opts.learning_rate = parameters['learning_rate']
    opts.ortho_init = False  # parameters['ortho_init']
    opts.logit_temp = parameters['logit_temp']
    opts.epochs = 10

    # run
    start_time = time.time()
    final_loss = train.train(opts)
    log_record.append(time.time() - start_time)
    log_record.append(final_loss)

    # complete trial
    if final_loss is None:
        print("ax trial", trial_index, "failed?")
        ax.log_trial_failure(trial_index=trial_index)
    else:
        ax.complete_trial(trial_index=trial_index,
                          raw_data={'final_loss': (final_loss, 0)})
    print("CURRENT_BEST", ax.get_best_parameters())

    # flush log
    log_msg = "\t".join(map(str, log_record))
    print(log_msg, file=log)
    print(log_msg)
    log.flush()
