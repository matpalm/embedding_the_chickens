# run faster_rcnn on all images in db that don't have detections yet.
# _highly_ recommended to set 'export TFHUB_CACHE_DIR=/data/tf_hub_module_cache/'
# note: initing the network and runs through are horrifically slow :/

from PIL import Image
import io
import sys
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from detections.detection import Detection
from tqdm import tqdm


class Detector(object):

    def __init__(self):
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        self.detector = hub.load(module_handle).signatures['default']

    def detections(self, fname):
        pil_image = Image.open(fname)
        w, h = pil_image.size

        network_input = np.array(pil_image, dtype=np.float32)
        network_input /= 255.0
        network_input = tf.convert_to_tensor(np.expand_dims(network_input, 0))

        result = self.detector(network_input)
        result = {key: value.numpy() for key, value in result.items()}

        boxes = result["detection_boxes"]
        class_names = result["detection_class_entities"]
        scores = result["detection_scores"]

        for box, class_name, score in zip(boxes, class_names, scores):
            if score > 0.1:
                ymin, xmin, ymax, xmax = tuple(box)
                x0, x1, y0, y1 = map(
                    int, [xmin * w, xmax * w, ymin * h, ymax * h])
                class_name = class_name.decode("ascii")
                yield Detection(-1, str(class_name), float(score), x0, y0, x1, y1)


if __name__ == "__main__":
    from db import img_db

    db = img_db.ImgDB()
    fnames_to_process = db.fnames_without_detections()
    if len(fnames_to_process) == 0:
        exit()

    # TODO: no doubt batching here would speed things up somewhat :/ i.e.
    # convert to tf.data pipeline but this model explicitly _doesn't_ support
    # batching :/ #faildog
    detector = Detector()
    for img_id, fname in tqdm(fnames_to_process):
        detections = detector.detections(fname)
        db.set_detections(img_id, detections)
