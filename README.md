# Staged Training method to ternarize weight for darknet and YOLO

reference papers
- [Training a Binary Weight Object Detector byKnowledge Transfer for Autonomous Driving](https://arxiv.org/pdf/1804.06332.pdf)  
- [Ternary weight networks](https://arxiv.org/pdf/1605.04711.pdf)  
- [XNOR-Net: ImageNet Classification Using BinaryConvolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)  

## Abstract  
Generally the inference task using full ternary weights -1,0,+1 is considered as low accuracy than full precision weights.  
But for mobile device such as raspberryPI small weights is efficiency choice  
Many quantization method for model weights are proposed now such as FP16, bfloat, fixed point 16bits, 8bit, ternary 4bits and XNor 1bit too.  
I consider that re-training after quatization of weights is needed.  
How to train using some quantization methods? from ground, finetune?  

I propose the staged training for yolov2-voc.cfg, yolov3-voc.cfg on [Darknet website](https://github.com/pjreddie).  
You can make full ternarized weights within 5 points accuracy drops for yolov2, yolov3 using this repository.  

Staged training generates Ternarized weights for yolov2-voc, yolov3-voc.  
Staged training method splits training step into 3 stages.  

Stage-0 : few layers are ternarized.  
Stage-1 : last some layers are ternarized.  
Stage-2 : all ayers without last layer are ternarized.  
Stage-3 : full ternarized.  

Weights used on each stages is imported from previous stage, such as stage-2 weights from stage-1.  

We trained yolov2-voc.cfg on 4 jobs, and checked each training curves.  

In fact, our experience denotes that accuracy drops about 5 points against full precision weights inferece. 

**result table regard to yolov2-voc.cfg**  

|Stage    |mAP  |IOU  |comments        |  
|-        |-    |-    |-               |  
|         |76.85|54.67|official weights|  
|M0       |77.09|57.04|                |  
|M1       |76.44|56.18|                |  
|M2       |75.06|57.71|                |  
|M3       |73.82|54.90|full ternary    |
|no staged|72.93|57.90|full ternary    |
Interation 41000(2000/class), steps 80%, 90%, lr=0.001  

**result table regard to yolov3-voc.cfg**  

|Stage    |mAP  |IOU  |comments        |  
|-        |-    |-    |-               |  
|         |75.54|62.78|FT form dn53.75 |
|M0       |75.02|63.04|                |  
|M1       |     |     |                |  
|M2       |     |     |                |  
|M3       |     |     |full ternary    |
|no staged|     |     |full ternary    |
Interation 100400(5000/class), steps 80%, 90%, lr=0.001  

## Staged training method for yolov2-voc.cfg  
1. prepare official weights that include full precision weights or train your model.cfg with VOC dataset.  
   I wget yolov2-voc.weights from pjreddie web site.  
   And prepare .data file too.  

2. create shell command file such as darknet.sh like below,  
   ./darknet detector train voc_M0.data yolov2-voc_M0.cfg yolov2-voc.weights  
   ./darknet detector train voc_M1.data yolov2-voc_M1.cfg backup_M0/yolov2-voc.minloss.weights  
   ./darknet detector train voc_M2.data yolov2-voc_M2.cfg backup_M1/yolov2-voc.minloss.weights  
   ./darknet detector train voc_M3.data yolov2-voc_M3.cfg backup_M2/yolov2-voc.minloss.weights  

3. go  
   ./darknet.sh

yolov2-voc.minloss.weigts file is saved at minimum loss.  
I spended 10days to get result of staged training yolov2 with GTX1080 x 2 environment.  

## More extra information regrad to this repo.  
- [segmenter](README_segmenter.md)  

