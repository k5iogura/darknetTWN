#!/usr/bin/env python3
import sys,os,json,argparse
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from io import StringIO

def check(f):
    assert os.path.exists(f),'not found {}'.fotmat(f)
    return str(f)
args=argparse.ArgumentParser()
args.add_argument('-r','--result_json',      type=check, required=True, help='anywhere/coco_results.json')
args.add_argument('-g','--groundtruth_json', type=check, required=True, help='anywhere/instance_val2014.json')
args=args.parse_args()

#path_to_annotation="/home/hst20076433/DATASET/coco/annotations/instances_val2014.json"
#path_to_results_dir="results"
#path_to_results_dir="results_4y"
resf = args.result_json
cocoGt = COCO(args.groundtruth_json)
cocoDt = cocoGt.loadRes(resf)
#resf = os.path.join(path_to_results_dir, 'instances_val2014_fakebbox100_results.json')
#cocoDt = cocoGt.loadRes(os.path.join(path_to_results_dir, 'instances_val2014_fakebbox100_results.json'))

with open(resf) as f:
    xx=json.load(f)
    ids=[i['image_id'] for i in xx]
    ids=sorted(list(set(ids)))

annType='bbox'
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = ids
cocoEval.evaluate()
cocoEval.accumulate()

original_stdout = sys.stdout
string_stdout = StringIO()
sys.stdout = string_stdout
cocoEval.summarize()
sys.stdout = original_stdout

mean_ap = cocoEval.stats[0].item()  # stats[0] records AP@[0.5:0.95]
detail = string_stdout.getvalue()

print(mean_ap)
print(detail)
