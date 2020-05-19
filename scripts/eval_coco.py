import sys,os,re,argparse
import pycocotools.coco as COCO
import pycocotools.cocoeval as COCOeval

def coco_bbox_eval(result_file, annotation_file):

    ann_type = 'bbox'
    coco_gt = COCO.COCO(annotation_file)
    coco_dt = coco_gt.loadRes(result_file)
    cocoevaler = COCOeval.COCOeval(coco_gt, coco_dt, ann_type)
    cocoevaler.evaluate()
    cocoevaler.accumulate()
    cocoevaler.summarize()

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-a', "--annotation_json", type=str, default=None)
    args.add_argument('-r', "--result_json"    , type=str, default=None)
    args = args.parse_args()

    coco_bbox_eval(args.result_json, args.annotation_json)

