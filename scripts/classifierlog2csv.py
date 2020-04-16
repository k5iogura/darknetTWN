#!/bin/env python
import re, os,sys,argparse

args = argparse.ArgumentParser()
args.add_argument('log', type=str)
args = args.parse_args()

with open(args.log) as f:
    for l in f:
        if re.search('avg',l) is None: continue
        items = l.strip().split(',')
        if not items[0].isdigit(): continue
        itera = int(items[0])
        epoch = float(items[1].split(':')[0])
        avg   = float(items[2].split()[0])
        print("{},{}".format(int(epoch),avg))
