#!/bin/env python3
import sys, os, re
import argparse
from pdb import set_trace
rejects = ['Girl', 'Woman', 'Tree', 'House', 'Building', 'Human leg', 'Man' , 'Boy','Human eye' ,'Human beard'
,'Human mouth' ,'Human body' ,'Human foot' ,'Human leg' ,'Human ear' ,'Human hair' ,'Human head'
,'Human face' ,'Human arm' ,'Human nose' ,'Human hand']


args = argparse.ArgumentParser()
args.add_argument('-c','--csv',type=str,default="train-annotations-bbox.csv",dest="filename")
args.add_argument('-m','--max',type=int,default=100000000)
args.add_argument('-o','--out',type=int,default=100000000)
args.add_argument('--thresh',type=float,default=0.3)
args.add_argument('-j','--jpeg_all',type=str,default="train.all.txt")
args.add_argument('-d','--class_desc',type=str,default="metadata/class-descriptions-boxable.csv")
args = args.parse_args()
max_line=args.max

jpegs = {}
with open(args.jpeg_all) as f:
    for l in f:
        imageid = os.path.splitext(os.path.basename(l.strip()))[0]
        jpegs.setdefault(imageid, l.strip())
descs = {}
with open(args.class_desc) as f:
    for l in f:
        LabelName,desc = l.strip().split(',')
        descs.setdefault(LabelName,[])
        descs[LabelName].append(desc)

image2annot={}
no=0
with open(args.filename,"r") as csv:
    # ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
    header=csv.readline().strip()
    assert header.split(',')[0] == 'ImageID',header
    print(header.split(','), file=sys.stderr)
    for no, l in enumerate(csv):
        comm=header+"=l.split(',')"
        exec(comm)
        iou = abs(float(XMax)-float(XMin))*abs(float(YMax)-float(YMin))
        image2annot.setdefault(ImageID, [])
        image2annot[ImageID].append([ LabelName, iou ])
        if no==max_line:break

class_file = '1annotation_images_class.csv'
class_set  = set()
for imgid in image2annot:
    if len(image2annot[imgid])==1 and image2annot[imgid][0][1]>args.thresh:
        classid = image2annot[imgid][0][0]
        if descs[classid][0] in rejects:continue
        class_set.add(classid)
        classID = re.sub('/m/','',classid)
        imgID = "{}_{}.jpg".format(imgid, classID)
        print('ln',jpegs[imgid],imgID)

with open(class_file,"w") as f:
    for classid in class_set:
        f.write("{},{}\n".format(classid,descs[classid][0]))
    print(len(classid),file=sys.stderr)

