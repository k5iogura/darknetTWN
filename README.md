# Ternary Weights for darknet and YOLO

reference papers
- Training a Binary Weight Object Detector byKnowledge Transfer for Autonomous Driving(https://arxiv.org/pdf/1804.06332.pdf)  
- Ternary weight networks(https://arxiv.org/pdf/1605.04711.pdf)  
- XNOR-Net: ImageNet Classification Using BinaryConvolutional Neural Networks(https://arxiv.org/pdf/1603.05279.pdf)  

## Abstract  
We applied the stage-wise training to yolov2-voc.cfg on Darknet website(https://github.com/pjreddie).  
Stage-wise training generate Ternarized weights for yolov2-voc.  
Stage-wise training splits training step into 3 stages.  

Stage-0 : few layers are ternarized.  
Stage-1 : last some layers are ternarized.  
Stage-2 : all ayers without last layer are ternarized.  
Stage-3 : full ternarized.  

Weights used on each stages is imported from previous stage, such as stage-2 weights from stage-1.  

We trained yolov2-voc.cfg into 4 jobs, and checked each training curves.  
Generary the inference with full ternary weights is considered as low accuracy than full precision weights.  

In fact our experience denote as same as above results but accuracy drop was about 3 points. 
