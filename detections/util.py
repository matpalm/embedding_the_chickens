import numpy as np


class AllowDenyFilter(object):

    def __init__(self, allow_list_str=None, deny_list_str=None):
        def split_str(list_str):
            return set([s.strip() for s in list_str.split(",")])

        if allow_list_str is None:
            self.allow_set = None
        else:
            self.allow_set = split_str(allow_list_str)

        if deny_list_str is None:
            self.deny_set = None
        else:
            self.deny_set = split_str(deny_list_str)

    def allow(self, s):
        if self.allow_set is not None:
            return s in self.allow_set
        if self.deny_set is not None:
            return s not in self.deny_set
        return True


def non_max_suppression(detections, overlap_thresh):

    if len(detections) == 0:
        return []

    # extract coordinates of the bounding boxes and scores from detections
    x1 = np.array([d.x0 for d in detections], dtype=np.float)
    y1 = np.array([d.y0 for d in detections], dtype=np.float)
    x2 = np.array([d.x1 for d in detections], dtype=np.float)
    y2 = np.array([d.y1 for d in detections], dtype=np.float)
    scores = np.array([d.score for d in detections])

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1e-10) * (y2 - y1 + 1e-10)
    idxs = np.argsort(scores)

    # { pick idxs: [ suppressions ] }
    pick_to_suppressions = {}

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:

        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have suppressed
        suppressed_idxs = np.concatenate(
            ([last], np.where(overlap > overlap_thresh)[0]))
        # ignore first since it's the same as i
        pick_to_suppressions[i] = list(reversed(idxs[suppressed_idxs[1:]]))
        idxs = np.delete(idxs, suppressed_idxs)

    # debug picks
    #print("pick_to_suppressions", pick_to_suppressions)

    # filter and return
    return [detections[p] for p in pick_to_suppressions.keys()]


def square_bb(x0, y0, x1, y1):
    # shuffle bounding box so that it's square, based on greater of
    # width or height, always centered around the middle of the crop
    cx, cy = (x0+x1)/2, (y0+y1)/2
    w, h = x1 - x0, y1 - y0
    if w > h:
        y0 = int(cy - w/2)
        y1 = int(cy + w/2)
    elif h > w:
        x0 = int(cx - h/2)
        x1 = int(cx + h/2)
    return x0, y0, x1, y1
