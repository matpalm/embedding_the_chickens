from embed_net import optimal_pairing
from embed_net import ensemble_net
import numpy as np\
    from detections import img_utils


def calculate_labels(logits_from_sims):
    # calc optimal pairing
    pairing = optimal_pairing.calculate(logits_from_sims)
    # convert to one hot
    labels = np.zeros_like(logits_from_sims)
    for i, j in pairing.items():
        labels[i, j] = 1.0
    return labels


crops_t0 = img_utils.load_crops_as_floats(
    'crops/pi_b/20200801/092957/crops.npy')
crops_t1 = img_utils.load_crops_as_floats(
    'crops/pi_b/20200801/093008/crops.npy')

params = ensemble_net.initial_params(num_models=10)
for _ in range(100):
    sims = ensemble_net.calc_sims(params, crops_t0, crops_t1)
    labels = calculate_labels(sims)
    params = ensemble_net.update(params, crops_t0, crops_t1, labels)
    print(ensemble_net.loss(params, crops_t0, crops_t1, labels))
