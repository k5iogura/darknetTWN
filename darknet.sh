#!/bin/bash
./darknet detector train voc_M0.data yolov2-voc_M0.cfg yolov2-voc.weights -clear -gpus 0,1 |& tee log_rW_v2-voc_M0
./darknet detector train voc_M1.data yolov2-voc_M1.cfg backup_rW_v2_M0/yolov2-voc_M0.backup -clear -gpus 0,1 |& tee log_rW_v2-voc_M1
./darknet detector train voc_M2.data yolov2-voc_M2.cfg backup_rW_v2_M1/yolov2-voc_M1.backup -clear -gpus 0,1 |& tee log_rW_v2-voc_M2
