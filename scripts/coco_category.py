#!/bin/env python
import cv2
import json
import numpy as np
import os
import random
import xml.dom.minidom
import xml.etree.cElementTree as et
from pdb import set_trace

#categories = {"1": [0, "person"], "2": [1, "bicycle"], "3": [2, "car"], "4": [3, "motorcycle"], "5": [4, "airplane"],
#              "6": [5, "bus"], "7": [6, "train"], "8": [7, "truck"], "9": [8, "boat"], "10": [9, "traffic light"],
#              "11": [10, "fire hydrant"], "13": [11, "stop sign"], "14": [12, "parking meter"], "15": [13, "bench"],
#              "16": [14, "bird"], "17": [15, "cat"], "18": [16, "dog"], "19": [17, "horse"], "20": [18, "sheep"],
#              "21": [19, "cow"], "22": [20, "elephant"], "23": [21, "bear"], "24": [22, "zebra"], "25": [23, "giraffe"],
#              "27": [24, "backpack"], "28": [25, "umbrella"], "31": [26, "handbag"], "32": [27, "tie"],
#              "33": [28, "suitcase"], "34": [29, "frisbee"], "35": [30, "skis"], "36": [31, "snowboard"],
#              "37": [32, "sports ball"], "38": [33, "kite"], "39": [34, "baseball bat"], "40": [35, "baseball glove"],
#              "41": [36, "skateboard"], "42": [37, "surfboard"], "43": [38, "tennis racket"], "44": [39, "bottle"],
#              "46": [40, "wine glass"], "47": [41, "cup"], "48": [42, "fork"], "49": [43, "knife"], "50": [44, "spoon"],
#              "51": [45, "bowl"], "52": [46, "banana"], "53": [47, "apple"], "54": [48, "sandwich"],
#              "55": [49, "orange"], "56": [50, "broccoli"], "57": [51, "carrot"], "58": [52, "hot dog"],
#              "59": [53, "pizza"], "60": [54, "donut"], "61": [55, "cake"], "62": [56, "chair"], "63": [57, "couch"],
#              "64": [58, "potted plant"], "65": [59, "bed"], "67": [60, "dining table"], "70": [61, "toilet"],
#              "72": [62, "tv"], "73": [63, "laptop"], "74": [64, "mouse"], "75": [65, "remote"], "76": [66, "keyboard"],
#              "77": [67, "cell phone"], "78": [68, "microwave"], "79": [69, "oven"], "80": [70, "toaster"],
#              "81": [71, "sink"], "82": [72, "refrigerator"], "84": [73, "book"], "85": [74, "clock"],
#              "86": [75, "vase"], "87": [76, "scissors"], "88": [77, "teddy bear"], "89": [78, "hair drier"],
#              "90": [79, "toothbrush"]}
categories = {}

dir_file = os.getcwd() + "/COCO"
dir_file = os.getcwd() + "/coco"
data_file = "train2014"

img_channel = 3
img_height = 256
img_width = 256
img_angle = 30

def save_label_shortname(dir_file):
    labels_list = "{}_names.list".format(dir_file)
    shortnames_list = "{}_shortnames.list".format(dir_file)
    with open(labels_list,"w") as labels_f:
        with open(shortnames_list,"w") as shorts_f:
            for key in categories:
                labels_f.write("%d\n"%int(categories[key][0]))
                shorts_f.write("%s\n"%str(categories[key][1]))
    print("write {} {} out".format(labels_list, shortnames_list))

def coco256x256(dir_file, data_file, is_train=True):
    """ Trimming 256x256 image from COCO dataset """
    instance_file = "%s/annotations/instances_%s.json" % (dir_file, data_file)

    dir_file256 = dir_file + "/coco256x256/"

    # make a directory to save trimming images
    if not os.path.exists(dir_file256 + data_file):
        os.makedirs(dir_file256 + data_file)

    # instances json file
    with open(instance_file, "rb") as file:
        dataset = json.load(file)

    annotations = dataset["annotations"]
    num_images  = len(dataset["images"])

    for i in range(len(dataset['categories'])):
        ctgid = dataset['categories'][i]['id']
        ctgnm = dataset['categories'][i]['name']
        categories[str(ctgid)] = [i,ctgnm]
    save_label_shortname(dir_file)

    filename_set = set()
    num_samples = 0
    img_mean = np.zeros((3, 224, 224), dtype="float32")  # (BGR, height, width)

    print("Processing {} {} ...".format(dir_file, data_file))
    with open("./" + data_file + "_coco256x256_map.txt", "w") as map_file:
        for ann in annotations:
            """
            segmentation : iscrowd=1 RLE(Run-length encoding) 
            area         : segmentation area
            iscrowd      : 0 or 1
            image_id     : image ID
            bbox         : [x-coordinate, y-coordinate, width, height]
            category_id  : category ID
            id           : ID
            """
            if ann["iscrowd"] == 1:
                continue
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            bbox = ann["bbox"]

            filename = "{:s}/images/{:s}/COCO_{:s}_{:0>12s}.jpg".format(dir_file, data_file, data_file, str(image_id))
            label = categories[str(category_id)][0]
            x, y, w, h = bbox

            #
            # trimming 256x256 image
            #
            img = cv2.imread(filename)
            try:
                if w < img_width / 4 or h < img_height / 4:  # remove very small bbox
                    continue
                elif h >= img_height and w >= img_width:  # width and height large bbox
                    img256 = cv2.resize(img[int(y): int(y + h), int(x): int(x + w), :], (img_width, img_height))
                elif w >= img_width and h < img_height:  # large width
                    yc = y + h / 2 - img_height / 2
                    if int(yc + img_height) >= img.shape[0]:  # outside of height
                        img256 = img[img.shape[0] - img_height:, int(x): int(x + w), :]
                    else:
                        img256 = img[int(yc): int(yc) + img_height, int(x): int(x + w), :]
                elif w < img_width and h >= img_height:  # large height
                    xc = x + w / 2 - img_width / 2
                    if int(xc + img_width) >= img.shape[1]:  # outside of width
                        img256 = img[int(y): int(y + h), img.shape[1] - img_width:, :]
                    else:
                        img256 = img[int(y): int(y + h), int(xc): int(xc) + img_width, :]
                else:
                    yc = y + h / 2 - img_height / 2
                    xc = x + w / 2 - img_width / 2
                    if int(yc + img_height) >= img.shape[0] and int(xc + img_width) >= img.shape[1]:
                        img256 = img[img.shape[0] - img_height:, img.shape[1] - img_width:, :]
                    elif int(yc + img_height) >= img.shape[0]:  # outside of height
                        img256 = img[img.shape[0] - img_height:, int(x): int(x + w), :]
                    elif int(xc + img_width) >= img.shape[1]:  # outside of width
                        img256 = img[int(y): int(y + h), img.shape[1] - img_width:, :]
                    else:
                        img256 = img[int(yc): int(yc) + img_height, int(xc): int(xc) + img_width, :]
                #
                # resize
                #
                img256 = cv2.resize(img256, (img_width, img_height))
                filename256 = "{:s}/{:s}/COCO_{:s}_{:0>12s}.jpg".format(dir_file256, data_file, data_file, str(image_id))

                while filename256 in filename_set:
                    filename256 = filename256[:-4] + str(random.randint(0, 9)) + '_' + str(image_id) + ".jpg"  # no overwrite
                filename_set.add(filename256)

            except cv2.error:
                continue

            #
            # rotation
            #
            if is_train:
                image_list = [img256]
                filename_list = [filename256[:-4] + '_' + str(label) + '.jpg']

                Ml = cv2.getRotationMatrix2D((img256.shape[1] // 2, img256.shape[0] // 2), angle=img_angle, scale=1.0)
                Mr = cv2.getRotationMatrix2D((img256.shape[1] // 2, img256.shape[0] // 2), angle=-img_angle, scale=1.0)

                channel_mean = img256.mean(axis=(0, 1))
                channel_value = int(channel_mean[0]), int(channel_mean[1]), int(channel_mean[2])

                image_list.append(cv2.warpAffine(img256, Ml, (img_width, img_height), borderValue=channel_value))
                image_list.append(cv2.warpAffine(img256, Mr, (img_width, img_height), borderValue=channel_value))

                filename_list.append(filename256[:-4] + "_l" + '_' + str(label) + ".jpg")
                filename_list.append(filename256[:-4] + "_r" + '_' + str(label) + ".jpg")

                for savename, saveimg in zip(filename_list, image_list):
                    map_file.write("%s\n" % (savename))
                    cv2.imwrite(savename, saveimg)
            else:
                map_file.write("%s\n" % (filename256))
                cv2.imwrite(filename256[:-4] + '_' + str(label) + '.jpg', img256)

            #
            # compute mean image
            #
            img_mean += img256[16:240, 16:240, :].transpose(2, 0, 1)

            num_samples += 1
            if num_samples % 10000 == 0:
                print("Now {} {}/{} samples...".format(data_file,num_samples,len(annotations)))

    #
    # save mean image as xml
    #
    if is_train:
        save_mean(data_file + "_coco256x256_mean.xml", img_mean / num_samples)
        np.save("coco21.npy", np.float32(img_mean / num_samples))

    #
    # complete trimming image
    #
    print("\nNumber of samples", num_samples)


def save_mean(mean_file, data):
    root = et.Element("opencv_storage")
    et.SubElement(root, "Channel").text = "3"
    et.SubElement(root, "Row").text = "224"
    et.SubElement(root, "Col").text = "224"

    img_mean = et.SubElement(root, "MeanImg", type_id="opencv-matrix")

    et.SubElement(img_mean, "rows").text = "1"
    et.SubElement(img_mean, "cols").text = str(3 * 224 * 224)
    et.SubElement(img_mean, "dt").text = "f"
    et.SubElement(img_mean, "data").text = " ".join(
        ["%e\n" % n if (i + 1) % 4 == 0 else "%e" % n for i, n in enumerate(np.reshape(data, (3 * 224 * 224)))])

    tree = et.ElementTree(root)
    tree.write(mean_file)
    x = xml.dom.minidom.parse(mean_file)
    with open(mean_file, "w") as f:
        f.write(x.toprettyxml(indent="  "))


if __name__ == "__main__":
    coco256x256(dir_file, "train2014", is_train=True)
    coco256x256(dir_file, "val2014"  , is_train=False)
    
