import re, os,sys,argparse
from pdb import set_trace

args = argparse.ArgumentParser()
args.add_argument('log', type=str)
args = args.parse_args()
log  = args.log

with open(log) as f:
    for line in f:
        line = line.strip()
        m = re.match(r'([0-9]+):',line)
        if m is None:continue
        batch = int(m.group().split(':')[0])
        loss  = re.search(r'([0-9.]+) avg,',line)
        avg = float(loss.group().split()[0])
        print('{},{}'.format(batch,avg))
