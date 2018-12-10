#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat ../data/intermediate/vocab.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1]\s" | cut -d' ' -f2 > ../data/intermediate/vocab_cut.txt
