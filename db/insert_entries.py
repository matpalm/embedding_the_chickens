#!/usr/bin/env python3

# insert entries for images into db
# open them in PIL as a first filter to check they are valid

# run daily as something like...
# find data/ -type f | grep 20180927 | sort | p db.insert_entries

import sys
from PIL import Image
from db import img_db


def fnames():
    for f in map(str.strip, sys.stdin.readlines()):
        try:
            Image.open(f)
            yield f
        except Exception as e:
            print("failed to open [%s] %s" % (f, str(e)))


db = img_db.ImgDB()
db.insert_fname_entries(list(fnames()))
