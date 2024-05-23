import os
import random
from pycocotools.coco import COCO
import json
import cv2
import argparse
from tqdm import tqdm

# cd c:/Users/a.sudupe/Desktop/TFG/Datasets/backup
# python ../../"Notebooks and Scripts"/split_train_test.py -i ./images_small_original -a c./annotations_all/COCO.json -d ./proba -s 0.7

def parse_args():
    parser = argparse.ArgumentParser(description='Make train test splits')
    parser.add_argument("-i", "--img",help="path to images")
    parser.add_argument("-m", "--mask",help="path to masks")
    parser.add_argument("-a", "--ann",help="path to coco annotations")
    parser.add_argument("-d", "--dest",help="path to dest folder")
    parser.add_argument("-s", "--size",help="train size (eg. 0.7). val and test will be the same size", default=0.7)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data_path = args.img
    mask_path = data_path + '/masks'
    img_path = data_path + '/images'
    anno_file_path = args.ann
    dest_path = args.dest

    ann_dest_path = os.path.join(dest_path + '/annotations')
    img_dest_path = os.path.join(dest_path + '/images')
    mask_dest_path = os.path.join(dest_path + '/masks')

    coco = COCO(anno_file_path)

    # path to destination folders
    train_img_folder = os.path.join(img_dest_path, 'train')
    val_img_folder = os.path.join(img_dest_path, 'val')
    test_img_folder = os.path.join(img_dest_path, 'test')

    train_mask_folder = os.path.join(mask_dest_path, 'train')
    val_mask_folder = os.path.join(mask_dest_path, 'val')
    test_mask_folder = os.path.join(mask_dest_path, 'test')

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png']

    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(img_path) if os.path.splitext(filename)[-1] in image_extensions]

    # Sets the random seed 
    random.seed(42)

    # Shuffle the list of image filenames
    random.shuffle(imgs_list)
    
    size = float(args.size)

    # determine the number of images for each set
    train_size = int(len(imgs_list) * size)
    val_size = int(len(imgs_list) * (1-size)/2)
    test_size = int(len(imgs_list) * (1-size)/2)

    # Create destination folders if they don't exist
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(ann_dest_path):
        os.makedirs(ann_dest_path)

    if not os.path.exists(img_dest_path):
        os.makedirs(img_dest_path)

    for folder_path in [train_img_folder, val_img_folder, test_img_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    for folder_path in [train_mask_folder, val_mask_folder, test_mask_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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

    data_list = coco.imgs.values()

    categories = [value for value in coco.cats.values()]
    images_test = []
    annotations_test = []
    images_train = []
    annotations_train = []
    images_val = []
    annotations_val = []

    def make_imgs(id, width, height, file_name):
        imgs = {'id': id, 'width': width, 'height': height, 'file_name': file_name}
        return imgs

    # Copy image files to destination folders
    for i, f in enumerate(tqdm(imgs_list)):

        id = get_id_from_file_name(f, data_list)

        if i < train_size:
            split_img_folder = train_img_folder
            split_mask_folder = train_mask_folder
            imgs = make_imgs(coco.imgs[id]['id'], coco.imgs[id]['width'], coco.imgs[id]['height'], coco.imgs[id]['file_name'])
            images_train.append(imgs)
            annotations_train.append(get_item_from_id(id, coco.anns.values()))
        elif i < train_size + val_size:
            split_img_folder = val_img_folder
            split_mask_folder = val_mask_folder
            imgs = make_imgs(coco.imgs[id]['id'], coco.imgs[id]['width'], coco.imgs[id]['height'], coco.imgs[id]['file_name'])
            images_val.append(imgs)
            annotations_val.append(get_item_from_id(id, coco.anns.values()))
        else:
            split_img_folder = test_img_folder
            split_mask_folder = test_mask_folder
            imgs = make_imgs(coco.imgs[id]['id'], coco.imgs[id]['width'], coco.imgs[id]['height'], coco.imgs[id]['file_name'])
            images_test.append(imgs)
            annotations_test.append(get_item_from_id(id, coco.anns.values()))
        
        image = cv2.imread(img_path + '/' + f)
        mask = cv2.imread(mask_path + '/' + f) #########################mask
        cv2.imwrite(split_img_folder + '/' + f, image)
        cv2.imwrite(split_mask_folder + '/' + f, mask)

    dataset_train = {'info': {"description":""}, 'images': images_train, 'annotations': annotations_train, "categories":categories}
    dataset_val = {'info': {"description":""}, 'images': images_val, 'annotations': annotations_val, "categories":categories}
    dataset_test = {'info': {"description":""}, 'images': images_test, 'annotations': annotations_test, "categories":categories}

    with open(ann_dest_path + r"\train.json", "w") as outfile: 
        json.dump(dataset_train, outfile)

    with open(ann_dest_path + r"\val.json", "w") as outfile: 
        json.dump(dataset_val, outfile)

    with open(ann_dest_path + r"\test.json", "w") as outfile: 
        json.dump(dataset_test, outfile)

if __name__ == "__main__":
    main()