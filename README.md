# Ternary Weights for darknet and YOLO

reference papers
- [Training a Binary Weight Object Detector byKnowledge Transfer for Autonomous Driving](https://arxiv.org/pdf/1804.06332.pdf)  
- [Ternary weight networks](https://arxiv.org/pdf/1605.04711.pdf)  
- [XNOR-Net: ImageNet Classification Using BinaryConvolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)  

## Abstract  
I propose the stage-wise training to yolov2-voc.cfg on [Darknet website](https://github.com/pjreddie).  
You can make full ternarized weights for yolov2 using this repository.  

Stage-wise training generate Ternarized weights for yolov2-voc.  
Stage-wise training splits training step into 3 stages.  

Stage-0 : few layers are ternarized.  
Stage-1 : last some layers are ternarized.  
Stage-2 : all ayers without last layer are ternarized.  
Stage-3 : full ternarized.  

Weights used on each stages is imported from previous stage, such as stage-2 weights from stage-1.  

We trained yolov2-voc.cfg on 4 jobs, and checked each training curves.  
Generary the inference with full ternary weights is considered as low accuracy than full precision weights.  

In fact our experience denote as same as above papaers results but accuracy drop was about 3 points. 

### stage-wise training  
1. prepare official weights that include full precision weights or train your model.cfg with VOC dataset.  
   I wget yolov2-voc.weights from pjreddie web site.  
2. make shell command file such as darknet.sh like below,  
   ./darknet detector train voc_M0.data yolov2-voc_M0.cfg yolov2-voc.weights  
   ./darknet detector train voc_M1.data yolov2-voc_M1.cfg backup_M0/yolov2-voc.minloss.weights  
   ./darknet detector train voc_M2.data yolov2-voc_M2.cfg backup_M1/yolov2-voc.minloss.weights  
   ./darknet detector train voc_M3.data yolov2-voc_M3.cfg backup_M2/yolov2-voc.minloss.weights  

yolov2-voc.minloss.weigts file is saved at minimum loss.  
I spended 10days to get result of stage-wise training yolov2 with GTX1080 x 2 environment.  
