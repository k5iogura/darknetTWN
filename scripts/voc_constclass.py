import os, sys, argparse
import re

args = argparse.ArgumentParser()
args.add_argument('train_txt',type=str)
args.add_argument('-px','--prefix',type=str, default='')
args = args.parse_args()

selects  = [ 1, 6, 8, 14, 18, 19 ]  # bicycle car chair person train tv

def choices(label,ids):
    anns = []
    with open(label) as f:
        for ann in f:
            tmp=list(map(float,ann.strip().split()))
            tmp[0] = int(tmp[0])
            if not tmp[0] in ids:continue
            anns.append(tmp)
    return anns

voc_names= [
'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

filename = args.train_txt if os.path.exists(args.train_txt) else sys.exit(-1)
names = [voc_names[i] for i in selects]

selects_histgram = [0]*len(selects)

cwd    = os.getcwd()
outnms = 'select%d_names.txt'%len(selects)
outjpg = 'select%d_train.txt'%len(selects)
lbldir = '%s/labels'%(cwd)
jpgdir = '%s/JPEGImages'%(cwd)
assert not os.path.exists(lbldir) ,lbldir
assert not os.path.exists(jpgdir) ,jpgdir

with open(filename) as f:
    contents = [i.strip() for i in f]
labels=[re.sub('jpg','txt',re.sub('JPEGImages','labels',i.strip())) for i in contents]
print("input file = {} contents={} labels={}".format(filename,len(contents),len(labels)))
print("selects ids = {} {}-classes".format(selects,len(selects)))
print(names)

labels_const = [os.path.join(lbldir,os.path.basename(i)) for i in labels]
jpegs_const  = [os.path.join(jpgdir,os.path.basename(i)) for i in contents]
print("makedirs",lbldir)
os.makedirs(lbldir,exist_ok=True)
print("makedirs",jpgdir)
os.makedirs(jpgdir,exist_ok=True)

Njpg = Nlbl = 0
with open(outjpg,"w") as jpgtxt:
    for jpg, label in enumerate(labels):
        annotates = choices(label, selects)
        if len(annotates)>0:
            ln_src = contents[jpg]
            ln_dst = jpegs_const[jpg]
            if not os.path.exists(ln_dst): os.symlink(ln_src, ln_dst)
            jpgtxt.write(jpegs_const[jpg]+"\n")
            Njpg+=1
            with open(labels_const[jpg],"w") as lbltxt:
                for ann in annotates:
                    new_id = selects.index(ann[0])
                    selects_histgram[new_id]+=1
                    ann[0] = new_id
                    lblstr = ' '.join(list(map(str,ann)))
                    lbltxt.write(lblstr+"\n")
                    Nlbl+=1
print("write out %s item names txt"%outnms)
with open(outnms, 'w') as nms:
    for nm in names:
        nms.write(str(nm)+'\n')

for i,h in enumerate(selects_histgram):print("{:4d} {:20s} {}".format(i,names[i],h))
print("write out jpeg list in {} {}-items {}-annotations".format(outjpg, Njpg, Nlbl))

