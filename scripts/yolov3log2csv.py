import re, os
from pdb import set_trace
log = 'log_v3-voc-ternary-s2off_stage1off_60550'
with open(log) as f:
    while True:
        line = f.readline().strip()
        m = re.match(r'([0-9]+):',line)
        if m is None:continue
        batch = int(m.group().split(':')[0])
        loss  = re.search(r'([0-9.]+) avg,',line)
        avg = float(loss.group().split()[0])
        print('{},{}'.format(batch,avg))
