
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
    def __init__(self, allow_deny_filter):
        self.db = img_db.ImgDB()
        self.colours = sorted(list(ImageColor.colormap.values()))
        self.c_idx = 0
        self.allow_deny_filter = allow_deny_filter

    def annotate_img(self, img_full_filename, min_score=0, show_all=False):
        img = Image.open(img_full_filename)

        detections = self.db.detections_for_img(img_full_filename)
        if len(detections) == 0:
            print("NOTE! no detections for %s" % img_full_filename)
            return img

        # collect all non filtered detections
        non_filtered_detections = [d for d in detections
                                   if self.allow_deny_filter.allow(d.entity)]
        if len(non_filtered_detections) == 0:
            print("NOTE! there were %d detections, but none after filtering" %
                  len(detections))
            return img

        # further filter through non_max_suppression, if not --show-all
        if show_all:
            picked_detections = non_filtered_detections
        else:
            picked_detections = u.non_max_suppression(
                non_filtered_detections, overlap_thresh=0.6)

        # draw detections on image and show
        canvas = ImageDraw.Draw(img, 'RGBA')
        font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
        for i, d in enumerate(picked_detections):
            if d.score < min_score:
                continue

            area = (d.x1-d.x0)*(d.y1-d.y0)
            alpha = 255
            entity_colour = ENTITY_TO_COLOUR.get(d.entity, (0, 0, 0))
            rectangle(canvas, xy=(d.x0, d.y0, d.x1, d.y1),
                      outline=(*entity_colour, alpha))

            debug_text = "e:%s: s:%0.3f a:%0.1f" % (d.entity, d.score, area)

            canvas.text(xy=(0, 25*i), text=debug_text, font=font, fill='black')
            canvas.text(xy=(1, 25*i+1), text=debug_text, font=font,
                        fill=entity_colour)
            print(debug_text)

        return img


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--entity-allow-list', type=str,
                        help='comma seperated list of entities to ignore')
    parser.add_argument('--entity-deny-list', type=str,
                        help='comma seperated list of entities to ignore')
    parser.add_argument('--min-score', type=float, default=0,
                        help='minimum detection score to show')
    parser.add_argument('--show-all', action='store_true',
                        help='show all detections (as opposed to nonmax suppressed)')
    parser.add_argument('--filename', type=str, default=None,
                        help="single file to show detections for")
    opts = parser.parse_args()
    print("opts %s" % opts, file=sys.stderr)

    allow_deny_filter = u.AllowDenyFilter(opts.entity_allow_list,
                                          opts.entity_deny_list)
    annotator = AnnotateImageWithDetections(allow_deny_filter)
    annotated_img = annotator.annotate_img(opts.filename,
                                           min_score=opts.min_score,
                                           show_all=opts.show_all)
    annotated_img.show()
