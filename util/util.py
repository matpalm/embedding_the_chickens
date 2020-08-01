import re

def split_fname(fname):
    m = re.match("data/pi_(.)/(\d*)/(\d*).jpg", fname)
    if not m:
        raise Exception("unparsable fname format [%s]" % fname)
    return m.groups()
