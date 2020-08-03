import re
import os


def split_fname(fname):
    m = re.match("data/(pi_.)/(\d*)/(\d*).jpg", fname)
    if not m:
        raise Exception("unparsable fname format [%s]" % fname)
    return m.groups()


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_exists_for_file(fname):
    ensure_dir_exists(os.path.dirname(fname))
