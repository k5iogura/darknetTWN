import cv2, os, sys
import numpy as np
from pdb import set_trace

def Truth2Relative(filename):

    IGNORE_IDs = ['DontCare', 'Misc']

    with open(filename) as f:
        ifiles = f.read().strip().split()

    img = cv2.imread(ifiles[0]) # All picture size are same in KITTI
    ih,iw,ic = [float(i) for i in img.shape]
    print('image size',ih,iw,ic)

    count=0
    obj_names = []
    for ifile in ifiles:
        loc_annotations  = []

        afile = ifile.replace('image_2', 'label_2')
        afile = afile.replace('.png', '.txt')
        with open(afile) as f:
            while True:
                annotate= f.readline().split()
                #
                # << format in KITTI annotation >>
                # [ object_id(str) truncation occlusion alpha left top right bottom 3D-box ]
                # occlusion : 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
                # alpha     : object observation angle ([-pi..pi])
                #
                if len(annotate)==0:break
                occlusion = int(annotate[2])
                if occlusion >= 2:continue # skip fully or unknown
                left, top, right, bottom = np.asarray([float(i) for i in annotate[4:8]])
                cx = (right + left)/iw/2.
                cy = (bottom + top)/ih/2.
                bw = (right - left)/iw
                bh = (bottom - top)/ih
                obj_id = annotate[0]
                if obj_id in IGNORE_IDs:continue
                if not obj_id in obj_names: obj_names.append( obj_id )
                idx = obj_names.index( obj_id )
                assert cx>=0 and cy>=0 and bw>=0 and bh>=0
                loc_annotations.append([idx, cx, cy, bw, bh])

        Afile = afile.replace('label_2', 'labels')
        Adir  = os.path.dirname(Afile)
        os.makedirs(Adir, exist_ok=True)
        with open(Afile, "w") as f:
            for loc in loc_annotations:
                loc_rep = '{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(loc[0],loc[1],loc[2],loc[3],loc[4])
                f.write(loc_rep)
            count+=1
            if (count % 500)==0:
                sys.stdout.write(str(count)+' ')
                sys.stdout.flush()
    return ifiles, obj_names

train_list = 'train.list'

print('Reading ', train_list, '...')
imageFiles, ObjNames = Truth2Relative(train_list)

print('annotation .txt files in', os.path.dirname(imageFiles[0]).replace('image_2','labels'),',',len(imageFiles),'files')
kitti_name = 'kitti.names'
print('label .names file as', kitti_name)
with open(kitti_name,'w') as f:
    for i in ObjNames:
        f.write('{}\n'.format(str(i)))

