#!/usr/bin/env bash
find data/ -type f | grep $1 | sort | python3 -m db.insert_entries