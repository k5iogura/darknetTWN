#!/bin/bash
sleep 129600
./darknet detector train voc_M0.data yolov2-voc_M0.cfg yolov2-voc.weights -clear -gpus 0,1 |& tee log_rW_v2-voc_M0
./darknet detector train voc_M1.data yolov2-voc_M1.cfg backup_rW_v2_M0/yolov2-voc_M0.minloss.weights -clear -gpus 0,1 |& tee log_rW_v2-voc_M1
./darknet detector train voc_M2.data yolov2-voc_M2.cfg backup_rW_v2_M1/yolov2-voc_M1.minloss.weights -clear -gpus 0,1 |& tee log_rW_v2-voc_M2
./darknet detector train voc_M3.data yolov2-voc_M3.cfg backup_rW_v2_M2/yolov2-voc_M2.minloss.weights -clear -gpus 0,1 |& tee log_rW_v2-voc_M3
