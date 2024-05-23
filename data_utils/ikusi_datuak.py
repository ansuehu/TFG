import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os
import cv2

coco = COCO(r'C:\Users\a.sudupe\Desktop\TFG\Datasets\backup\Proba_osoa\patched\annotations\coco.json')

img_ids = list(coco.imgs.keys())

for idx in img_ids:

    image_info = coco.loadImgs(idx)[0]
    mask_path = r'C:\Users\a.sudupe\Desktop\TFG\Datasets\backup\Proba_osoa\patched/masks/' + image_info['file_name']
    mask = cv2.imread(mask_path)
    seg = coco.loadAnns(idx)[0]['segmentation'][0]
    print(image_info['file_name'])

    x = []
    y = []
    for i in range(len(seg)):
        if i%2==0: x.append(seg[i])
        else: y.append(seg[i])

    plt.imshow(mask)
    
    if len(x)>0: plt.scatter(x, y)
    plt.show()
