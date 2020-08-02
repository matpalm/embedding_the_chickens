#!/usr/bin/env python3
from db import img_db
from PIL import Image, ImageDraw, ImageColor, ImageFont
import numpy as np
import detections.util as u

ENTITY_TO_COLOUR = {
    'Chicken': (255, 0, 0),
    'Animal': (0, 255, 0),
    'Bird': (0, 0, 255),
    'Duck': (255, 255, 0)
}


def rectangle(canvas, xy, outline, width=5):
    x0, y0, x1, y1 = xy
    corners = (x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)
    canvas.line(corners, fill=outline, width=width)


class AnnotateImageWithDetections(object):
    def __init__(self, entities_blacklist=[]):
        self.db = img_db.ImgDB()
        self.colours = sorted(list(ImageColor.colormap.values()))
        self.c_idx = 0
        self.entities_blacklist = set(entities_blacklist)

    def annotate_img(self, img_full_filename, min_score=0, show_all=False):
        img = Image.open(img_full_filename)

        detections = self.db.detections_for_img(img_full_filename)
        if len(detections) == 0:
            print("NOTE! no detections for %s" % img_full_filename)
            return img

        # collect all detections
        bounding_boxes = []
        entities = []
        scores = []
        for d in detections:
            if d.entity in self.entities_blacklist:
                continue
            bounding_boxes.append([d.x0, d.y0, d.x1, d.y1])
            entities.append(d.entity)
            scores.append(d.score)
        if len(bounding_boxes) == 0:
            print("NOTE! there were %d detections, but none after filtering" %
                  len(detections))
            return img

        # stack all detections
        bounding_boxes = np.stack(bounding_boxes)
        entities = np.array(entities)
        scores = np.array(scores)
        # print("bounding_boxes", bounding_boxes)
        # print("entities", entities)
        # print("scores", scores)

        # decide which detections to pick; either all of them (for show_all)
        # or by doing non max suppression to collect key bounding boxes along
        # with their corresponding suppressions
        if show_all:
            picks = range(len(bounding_boxes))
        else:
            pick_to_suppressions = u.non_max_suppression(
                bounding_boxes, scores=scores, overlap_thresh=0.6)
            print("pick_to_suppressions", pick_to_suppressions)
            picks = pick_to_suppressions.keys()

        # draw detections on image and show
        canvas = ImageDraw.Draw(img, 'RGBA')
        font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        for i, pick in enumerate(picks):
            score = scores[pick]
            if score < min_score:
                continue

            bounding_box = bounding_boxes[pick]
            entity = entities[pick]

            x0, y0, x1, y1 = bounding_box
            area = (x1-x0)*(y1-y0)
            alpha = 255
            entity_colour = ENTITY_TO_COLOUR.get(entity, (0, 0, 0))
            rectangle(canvas, xy=(x0, y0, x1, y1),
                      outline=(*entity_colour, alpha))

            if show_all:
                debug_text = "e:%s: s:%0.3f a:%0.1f  p:%d" % (
                    entity, score, area, pick)
            else:
                suppressions = pick_to_suppressions[pick]
                debug_text = "e:%s: s:%0.3f a:%0.1f  p:%d sup:%s" % (
                    entity, score, area, pick, suppressions)

            canvas.text(xy=(0, 25*i), text=debug_text, font=font, fill='black')
            canvas.text(xy=(1, 25*i+1), text=debug_text,
                        font=font, fill=entity_colour)
            print(debug_text)

        return img


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--entity-blacklist', type=str, default='',
                        help='comma seperated list of entities to ignore')
    parser.add_argument('--min-score', type=float, default=0,
                        help='minimum detection score to show')
    parser.add_argument('--show-all', action='store_true',
                        help='show all detections (as opposed to nonmax suppressed)')
    parser.add_argument('--filename', type=str, default=None,
                        help="single file to show detections for")
    opts = parser.parse_args()
    print("opts %s" % opts, file=sys.stderr)

    annotator = AnnotateImageWithDetections(
        entities_blacklist=opts.entity_blacklist.split(","))
    annotated_img = annotator.annotate_img(opts.filename,
                                           min_score=opts.min_score,
                                           show_all=opts.show_all)
    annotated_img.show()
