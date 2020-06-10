# scripts 

### training log viewer : avgout.py  

Shows progress bar via training log.  

```$ avgout.py [log-file]  
CPU: Linux/v100/3.10.0-693.21.1.el7.x86_64
CWD: darknetTWN
phrase to search iteration and loss: ([0-9]+):.* +([0-9.]+) *avg
loss range: 257.627747 to 3.323541
-----------------------------------------------------------------------------------------------------------
|  iteration         loss                                       bar                                       |
-----------------------------------------------------------------------------------------------------------
          97   217.604782 ################################################################################|
         194    52.951962 ####################################################################
         291    29.711544 ######################################
         388    22.803951 #############################
         485    14.816415 ###################
         582    10.214262 #############
         679     6.632885 #########
         776     5.716087 ########
         873     4.426070 ######
         970     4.172388 ######
        1067     4.087748 ######
        1164     4.039390 ######
        1261     3.746985 #####
        1358     3.849923 #####
        1455     3.702813 #####
        1552     3.614410 #####
        1649     3.555058 #####
        1746     3.643238 #####
        1843     3.426941 #####
        1940     3.567107 #####
-----------------------------------------------------------------------------------------------------------
MIN     1845     3.323541 >>>>>
-----------------------------------------------------------------------------------------------------------
```  

Easy way to watch progress countinuosly,  
`$ watch -n 5 './scripts/avgout.py [log-file]'`  

### OpenImage downloader : oi_download.sh oi_label.sh  

`$ mkdir work; cd work`  
`$ ./oi_download.sh`  
   unzip `*`.zip files  
`$ find $(pwd) -iname \*.jpg > train.all.txt`  
`$ ./oi_labels.sh --thresh 0.3 > ln.sh`  
`$ mkdir JPEGImages ; cd JPEGImages`  
`$ bash ../ln.sh`  
`$ ..; find $(pwd)/JPEGImages -iname \*.jpg | sort -R >train.txt`  

downloads `*`.csv, `*`/zip.  
selects image ocuppied over 30% and makes symbolic links in JPEGImages directory.  

### COCO Downloader : get_cococ_download.py  

`$ ./get_coco_download.py`  

see ./coco directory.  

### Evaluation mAP with pycocotools : pycocoeval.py  

Notice : pycocotools is reuired. see [pycocotools](https://github.com/cocodataset/cocoapi) to install instructions.  
git clone https://github.com/cocodataset/cocoapi  
For Python, run "make" under coco/PythonAPI  
python3 -c "import pycocotools" # for checking installation  

```
 $ ./scripts/pycocoeval.py -r results/coco_results.json -g anywhere/coco/annotations/instances_val2014.json
```

Maybe nessesary python3-tkinter, matplotlib==2.1.0 etc.  

### Transform darknet VOC format to COCO json format : voc2json.py  

From darknet VOC labels .txt and voc.names file to .json for pycocotools  
```
 $ voc2json -n data/voc.names -i anywhere/VOC/2007_test.txt -j 2007_test.json
```
See ./2007_test.json for pycocotools.  
