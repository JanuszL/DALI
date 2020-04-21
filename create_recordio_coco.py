import json
import os
import sys
import random
from collections import defaultdict
import mxnet as mx

in_path = sys.argv[1]
in_img_path = sys.argv[2]
out_path = sys.argv[3]
shuffle = False
if len(sys.argv) > 4:
    shuffle = True

print("Annotations file (with path): {}".format(in_path))
print("Images path (directory): {}".format(in_img_path))
print("Output path path (directory): {}".format(out_path))
print("Shuffling [True/False]: {}".format(shuffle))

dataset = json.load(open(in_path, 'r'))

anns, imgs, cats = {}, {}, {}
imgToAnns = defaultdict(list)

if 'annotations' in dataset:
    for ann in dataset['annotations']:
        imgToAnns[ann['image_id']].append(ann)
        anns[ann['id']] = ann

if 'images' in dataset:
    for img in dataset['images']:
        imgs[img['id']] = img

if 'categories' in dataset:
    for i, cat in enumerate(dataset['categories']):
        cats[cat['id']] = i + 1

fname_rec = os.path.join(out_path, 'coco.rec')
fname_idx = os.path.join(out_path, 'coco.idx')
record = mx.recordio.MXIndexedRecordIO(fname_idx, fname_rec, 'w')

# create RecordIO
images = dataset['images']
if shuffle:
    random.shuffle(images)
for i, img in enumerate(images):
    filePath = img["file_name"]
    with open(os.path.join(in_img_path, filePath), 'rb') as fin:
        imgData = fin.read()
    imgID = img["id"]

    imgWidth = img['width']
    imgHeight = img['height']

    bboxes = []
    labels = []
    for ann in imgToAnns[imgID]:
        bboxes.append(ann["bbox"])
        labels.append(cats[ann["category_id"]])
    objNum = len(bboxes)

    mergedBboxes = []
    for bbox in bboxes:
        mergedBboxes += bbox

    finalData = [imgID, imgWidth, imgHeight] + labels + mergedBboxes

    header = mx.recordio.IRHeader(0, finalData, i, 0)
    s = mx.recordio.pack(header, imgData)
    record.write_idx(i, s)
    if i % 100 == 0:
        print(i)







