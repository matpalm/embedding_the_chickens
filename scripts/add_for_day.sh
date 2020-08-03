#!/usr/bin/env bash

if [ -z "$1" ]; then
  echo "usage: $0 YMD"
  exit -1
fi

find data/ -type f | grep $1 | sort | python3 -m db.insert_entries