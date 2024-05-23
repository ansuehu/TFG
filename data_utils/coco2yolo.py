import os
from pycocotools.coco import COCO
import shutil
import cv2
import argparse
from tqdm import tqdm
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Change format from COCO to YOLO')
    parser.add_argument("-i", "--img",help="path to coco annotations")
    parser.add_argument("-d", "--dest",help="path to yolo folder")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dir_path = args.img
    dest_path = args.dest
    images_path = dir_path + '/images'
    anno_path = dir_path + "/annotations"

    # path to destination folders
    train_folder = dest_path + '/train'
    val_folder = dest_path + '/val'
    test_folder = dest_path + '/test'

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Create destination folders if they don't exist
    for folder_path in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            os.makedirs(os.path.join(folder_path, 'images'))
            os.makedirs(os.path.join(folder_path, 'labels'))
        elif not os.path.exists(os.path.join(folder_path, 'images')) and not os.path.exists(os.path.join(folder_path, 'labels')):
            os.makedirs(os.path.join(folder_path, 'images'))
            os.makedirs(os.path.join(folder_path, 'labels'))

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

    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Create a list of image filenames in 'data_path'
    imgs_list = {f:[filename for filename in os.listdir(os.path.join(images_path, f)) if os.path.splitext(filename)[-1] in image_extensions] for f in os.listdir(images_path)}

    folders = imgs_list.keys()

    for f in folders:

        path = os.path.join(dest_path, f)
        data_path_f =  os.path.join(images_path, f)
        coco = COCO(anno_path + '/{}.json'.format(f))

        for i, image in enumerate(tqdm(imgs_list[f])):
        
            shutil.copy(os.path.join(data_path_f, image), os.path.join(path, 'images' ,image))

            # imagecv = cv2.imread(os.path.join(data_path_f, image))
            # imagecv = cv2.resize(imagecv, (640, 640))
            # cv2.imwrite(os.path.join(path, 'images' ,image), imagecv)

            name = os.path.splitext(image)[0]

            id = get_id_from_file_name(image, coco.imgs.values())

            txt = str(coco.anns[id]['category_id'])
            for j in range(0, len(coco.anns[id]['segmentation'][0]), 2):
                
                x = int(get_item_from_id(id, coco.anns.values())['segmentation'][0][j])/int(coco.imgs[id]['width']-1)
                y = int(get_item_from_id(id, coco.anns.values())['segmentation'][0][j+1])/int(coco.imgs[id]['height']-1)
                txt += " "+ str(x) + " " + str(y)

            f = open(path + '/labels/' + name + '.txt', "w")
            f.write(txt)
            f.close()
            # print(image)

    data = dict(
        train = './train/images',
        val = './val/images',
        test = './test/images',
        nc = 1,
        names = ['Tableroa']
    )

    with open(dest_path+'/data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)




if __name__ == "__main__":
    main()