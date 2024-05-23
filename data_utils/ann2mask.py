from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Create masks from annotations')
    parser.add_argument("-i", "--img",help="path to data")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_path = args.img
    data_path = data_path.replace("\\", '/')
    ann_path = data_path + '/annotations'
    mask_path = data_path + '/masks'
    

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    
    try:
        coco = COCO(ann_path + '/COCO.json')
    except():
        return 1
    # image_ids = np.random.choice(dataset.image_ids, 3)
    images = coco.imgs
    for img_id in tqdm(images):
        img = images[img_id]
        # image = Image.open(img_path + img['file_name'])
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = 255*coco.annToMask(anns[0])
        # for i in range(len(anns[1:])):
        #     mask += (254-i)*coco.annToMask(anns[i])

        im = Image.fromarray(mask)

        im = im.convert('L')
        im = im.point(lambda p: 255 if p>200 else 0)
        # print(np.unique(im))
        

        im.save(mask_path + '/' + img['file_name'], quality=100, subsampling=0)
    
    return 0

if __name__ == "__main__":
    main()