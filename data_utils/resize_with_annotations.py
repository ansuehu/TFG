import cv2
import os
from pycocotools.coco import COCO
import json

anno_path = r'C:\Users\a.sudupe\Desktop\TFG\Datasets\Argazkiak_COCO\annotations/'
img_path = r'C:\Users\a.sudupe\Desktop\TFG\Datasets\Argazkiak_COCO\images/'

dest_path = r'C:\Users\a.sudupe\Desktop\TFG\Datasets\Argazkiak_COCO_resized/'

paths = ['train', 'test', 'val']
r = 1024

def get_id_from_file_name(file_name, data_list):
    for item in data_list:
        if item['file_name'] == file_name:
            return item['id']
    return None

def get_item_from_id(id, data_list):
    for item in data_list:
        if item['image_id'] == id:
            return item
    return None

def make_imgs(id, width, height, file_name):
    imgs = {'id': id, 'width': width, 'height': height, 'file_name': file_name}
    return imgs

def make_anns(id, iscrowd, image_id, category_id, segmentation, bbox, area):
    return {'id': id, 'iscrowd': iscrowd, 'image_id': image_id, 'category_id': category_id, 'segmentation': [segmentation], 'bbox': bbox, 'area': area}
    
for p in paths:
    path = img_path + p
    imgs = os.listdir(path)
    coco = COCO(anno_path + p + '.json')
    categories = [value for value in coco.cats.values()]
    
    images = []
    annotations = []

    for im in imgs:
        image = cv2.imread(path + '/' + im)
        image = cv2.resize(image, (r, r))
        cv2.imwrite(dest_path + 'images/' + p + '/' + im, image)

        id = get_id_from_file_name(im, coco.imgs.values())

        imgs = make_imgs(coco.imgs[id]['id'], r, r, coco.imgs[id]['file_name'])
        images.append(imgs)
        anns = get_item_from_id(id, coco.anns.values())
        seg = []
        bbox = []
        for a in range(0, len(anns['segmentation'][0]), 2):
            seg.append(anns['segmentation'][0][a]*r/int(coco.imgs[id]['width']))
            seg.append(anns['segmentation'][0][a+1]*r/int(coco.imgs[id]['height']))

        for a in range(0, len(anns['bbox']), 2):
            bbox.append(anns['bbox'][a]*r/int(coco.imgs[id]['width']))
            bbox.append(anns['bbox'][a+1]*r/int(coco.imgs[id]['height']))

        anns = make_anns(anns['id'], anns['iscrowd'], anns['image_id'], anns['category_id'], seg, bbox, 0)
        annotations.append(anns)
    

    dataset = {'info': {"description":""}, 'images': images, 'annotations': annotations, "categories": categories}

    with open(dest_path + 'annotations/' + p + '.json', "w") as outfile: 
        json.dump(dataset, outfile, indent=6)



