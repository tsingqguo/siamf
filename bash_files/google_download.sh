#!/usr/bin/env bash
# https://drive.google.com/file/d/1e51IL1UZ-5seum2yUYpf98l2lJGUTnhs/view?usp=sharing
FILEID=$1
FILENAME=$2
wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \
    --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" \
    -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
