#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "usage: $0 YMD"
  exit -1
fi

find data/ -type f -iname *jpg | grep $1 | sort > manifest.$1
python3 -m detections.extract_crops \
 --manifest manifest.$1 \
 --entity-allow-list Chicken,Bird,Animal \
 --min-score 0.2
rm manifest.$1