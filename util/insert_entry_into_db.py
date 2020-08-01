#!/usr/bin/env python3

# insert entries for images into db

# run daily as something like...
# find data/ -type f | grep 20180927 | sort | p util.insert_entry_into_db

from util import img_db
import sys

db = img_db.ImgDB()

fnames = [f.strip() for f in sys.stdin.readlines()]
db.insert_fname_entries(fnames)
