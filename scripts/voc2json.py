import sys,os,re,argparse
import numpy as np
import json,cv2

#TOP# [u'info', u'images', u'licenses', u'annotations', u'categories']
#CAT# [{u'supercategory': u'person', u'id': 1, u'name': u'person'}, ...
#IMG# {u'license': 3, u'file_name': u'COCO_val2014_000000391895.jpg', u'coco_url': u'http://mscoco.org/images/391895', u'height': 360, u'width': 640, u'date_captured': u'2013-11-14 11:18:45', u'flickr_url': u'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg', u'id': 391895}
#ANN#{u'segmentation': [[239.97, 260.24, 222.04, 270.49, 199.84, 253.41, 213.5, 227.79, 259.62, 200.46, 274.13, 202.17, 277.55, 210.71, 249.37, 253.41, 237.41, 264.51, 242.54, 261.95, 228.87, 271.34]], u'area': 2765.1486500000005, u'iscrowd': 0, u'image_id': 558840, u'bbox': [199.84, 200.46, 77.71, 70.88], u'category_id': 58, u'id': 156}

def check(f):
    assert os.path.exists(f),f
    return str(f)
args=argparse.ArgumentParser()
args.add_argument('-i','--in_list',  type=check,default='2007_test.txt')
args.add_argument('-n','--names',    type=check,default='voc.names')
args.add_argument('-j','--json_file',type=str,default='voc_2007_test.json')
args=args.parse_args()

print(args)
with open(args.names) as f:
    names = [i.strip() for i in f]
print("{} categories".format(len(names)))

with open(args.in_list) as f:
    voc_jpg_list = [i.strip() for i in f]
    voc_ann_list = [re.sub('.jpg','.txt',re.sub('JPEGImages','labels',i.strip())) for i in voc_jpg_list]
assert len(voc_jpg_list)==len(voc_ann_list)
assert os.path.exists(voc_jpg_list[0]) and os.path.exists(voc_ann_list[0])

# INFO
inf = {u'description': u'This is 2014 VOC dataset.', u'url': u'http://voc.org', u'version': u'1.0', u'year': 2014, u'contributor': '', u'date_created': u'2015-01-27 09:11:52.357475'}

# LICENSES
lic = [{u'url': u'http://creativecommons.org/licenses/by-nc-sa/2.0/', u'id': 1, u'name': u'Attribution-NonCommercial-ShareAlike License'}, {u'url': u'http://creativecommons.org/licenses/by-nc/2.0/', u'id': 2, u'name': u'Attribution-NonCommercial License'}, {u'url': u'http://creativecommons.org/licenses/by-nc-nd/2.0/', u'id': 3, u'name': u'Attribution-NonCommercial-NoDerivs License'}, {u'url': u'http://creativecommons.org/licenses/by/2.0/', u'id': 4, u'name': u'Attribution License'}, {u'url': u'http://creativecommons.org/licenses/by-sa/2.0/', u'id': 5, u'name': u'Attribution-ShareAlike License'}, {u'url': u'http://creativecommons.org/licenses/by-nd/2.0/', u'id': 6, u'name': u'Attribution-NoDerivs License'}, {u'url': u'http://flickr.com/commons/usage/', u'id': 7, u'name': u'No known copyright restrictions'}, {u'url': u'http://www.usa.gov/copyright.shtml', u'id': 8, u'name': u'United States Government Work'}]

# CATEGORIES
print('#CATEGORIES...')
categories=[]
for i,c in enumerate(names):
    category={}
    category.setdefault( 'supercategory'  , str(c))
    category.setdefault( 'id'             , i+1 )
    category.setdefault( 'name'           , str(c) )
    categories.append(category)
print('#done.')

def get_image_id(j):
    image_id = os.path.splitext(os.path.basename(j))[0]
    image_id = int(re.sub('^.+_','',image_id))
    return image_id

# IMAGES
print('#IMAGES...')
images=[]
for i,j in enumerate(voc_jpg_list):
    image ={}
    file_name = os.path.basename(j)
    image_id = get_image_id(j)
    img = cv2.imread(j)
    h,w = img.shape[:2]
    image.setdefault( 'license'  ,3 )
    image.setdefault( 'file_name',file_name )
    image.setdefault( 'coco_url' ,"" )
    image.setdefault( 'height'   ,h )
    image.setdefault( 'width'    ,w )
    image.setdefault( 'id'       ,image_id )
    images.append(image)
print('#done.')

# ANNOTATIONS
print('#ANNOTATIONS...')
anns= []
aid = 0
for i,j in enumerate(voc_jpg_list):
    a = voc_ann_list[i]
    image_id = get_image_id(j)
    with open(a) as f:
        H,W = images[i]['height'], images[i]['width']
        for A in f:
            ann = {}
            cid,x,y,w,h = A.strip().split()
            x,y,w,h = int(W*float(x)), int(H*float(y)), int(W*float(w)), int(H*float(h))
            ann.setdefault( 'segmentation'  ,[] )
            ann.setdefault( 'area'          ,float(w*h) )
            ann.setdefault( 'iscrowd'       ,0 )
            ann.setdefault( 'image_id'      ,image_id )
            ann.setdefault( 'bbox'          ,[x,y,w,h])
            ann.setdefault( 'category_id'   ,int(cid))
            ann.setdefault( 'id'            ,int(aid))
            aid += 1
            anns.append(ann)
print('#done.')

#TOP [u'info', u'images', u'licenses', u'annotations', u'categories']
print('#TOP...')
ds = {}
ds.setdefault('info', inf)
ds.setdefault('images', images)
ds.setdefault('licenses', lic)
ds.setdefault('annotations', anns)
ds.setdefault('categories', categories)

print("{} json.dump ...".format(args.json_file))
with open(args.json_file,'w') as f:
    json.dump(ds,f,indent=4)
print("{} generated".format(args.json_file))
