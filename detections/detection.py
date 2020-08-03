from collections import namedtuple

Detection = namedtuple(
    'Detection', ['id', 'entity', 'score', 'x0', 'y0', 'x1', 'y1'])
