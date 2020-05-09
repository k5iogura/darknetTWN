#!/bin/env python3
import re,sys,os,argparse
import numpy as np
from math import ceil
from pdb import set_trace

args = argparse.ArgumentParser()
args.add_argument('log', type=str)
args.add_argument('-p', "--phrase",   type=str,default="detv3", choices=["detv3","class","seg"])
args.add_argument('-c', "--column",   type=int,default=20)
args.add_argument('-s', "--stop_iter",type=int,default=1000000)
args.add_argument('-b', "--bars",     type=int,default=80)
args = args.parse_args()
filename=args.log

#
# setup searching phrase in log files
# append phrase for new log format
#
if args.phrase == "detv3":
    reg = '([0-9]+):.* +([0-9.]+) *avg'
else:
    print(args.phrase, "Not suported")
    sys.exit(-1)
print("CPU:","/".join(os.uname()[i] for i in range(3)))
print("CWD:",os.getcwd())
print('phrase to search iteration and loss:',reg)

#
# analyzing
#
iter2avg = [[0,1e-10]]
with open(filename) as f:
    for l in f:
        w = l.strip().split()
        if len(w)==0:continue

        m = re.search(reg,l.strip())

        if m is None or len(m.groups())<2:continue
        iterNo, loss = m.groups()[:2]
        iterNo, loss = int(iterNo), float(loss)
        iter2avg.append([ iterNo, loss ])
        if iterNo > args.stop_iter:break
iter2avg = np.asarray(iter2avg)

N=len(iter2avg)
Offset=int(N/args.column)
if Offset<=0:sys.exit(-1)
maximumj = max(iter2avg[0:,1])
minimumj = min(iter2avg[1:,1])
minimumi = [int(i) for i,loss in iter2avg if loss == minimumj][-1] # avoid multi-hit
print('loss range:',maximumj,'to',minimumj)
minimumj = minimumj if minimumj > 0.0 else 1e-10

#
# show results
#
print("-"*(args.bars+27))
fstr = "|{:>11s} {:>12s} {:^%ds}|"%(args.bars)
print(fstr.format("iteration","loss","bar"))
print("-"*(args.bars+27))

maximumj = np.max(iter2avg[::Offset,1])
maximumj = min(maximumj, args.column*minimumj)
limit = 10
widen = 1.0
if maximumj<limit*minimumj:
    widen = (limit*minimumj)/maximumj
    maximumj=limit*minimumj
for i,loss in iter2avg[::Offset]:
    if i == 0: continue
    wloss = widen * loss
    sys.stdout.write("{:12d} {:12.6f} ".format(int(i),loss))
    for j in range(ceil(args.bars*wloss/(maximumj-minimumj))):
        if j > args.bars:continue
        if j == args.bars:
            sys.stdout.write('|')
        else:
            sys.stdout.write('#')
    print('')
print("-"*(args.bars+27))
sys.stdout.write("MIN {:8d} {:12.6f} ".format(minimumi,minimumj))
for j in range(ceil(args.bars*widen*minimumj/(maximumj-minimumj))):
    if j > args.bars:break
    sys.stdout.write('>')
print('')
print("-"*(args.bars+27))

