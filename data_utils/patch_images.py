import os
from patchify import patchify
import cv2
from tqdm import tqdm
# import annotation_helper as ah
import numpy as np
import json        
from scipy.spatial import distance
import argparse
# import annotation_helper as ah


def parse_args():
    parser = argparse.ArgumentParser(description='Make smaller patches')
    parser.add_argument("-i", "--images",help="path to images")
    parser.add_argument("-d", "--dest",help="path to dest folder")
    parser.add_argument("-s", "--size",help="Size of the images")
    parser.add_argument("-o", "--overlap",help="Overlap")

    args = parser.parse_args()
    return args

# Emandako puntua baño ezkerrerago dauden puntuak itzultzen ditu, ordenatuta
def ezkerreko_gertuena(points, p):
    min = (0,0)
    min_dist = 1000000
    min_index = None
    for i, point in enumerate(points):
        dist = distance.euclidean(p, point)
        if point[0]<=p[0] and min_dist>dist:
            min = point
            min_index = i
            min_dist = dist
    if min_dist == 1000000:
        return None
    return min, min_index

# Emandako puntua baño eskubirago dauden puntuak itzultzen ditu, ordenatuta
def eskuineko_gertuena(points, p, img):
    min = (img.shape[0]-1,0)
    min_dist = 1000000
    min_index = None, None
    for i, point in enumerate(points):
        # print(point)
        dist = distance.euclidean(p, point)
        if point[0]>=p[0] and min_dist>dist:
            min = point
            min_index = i
            min_dist = dist

    if min_dist == 1000000:
        return None, None
    return min, min_index

# Emandako 2D-ko lista, erlojuaren zentzuan ordenatzen du
def ordenatu_lista(seg, img):

    seg_ordenatuta = []
    s = (img.shape[0]-1, len(img[0])-1)
    ezker = ezkerreko_gertuena(seg, s)
    if ezker == None:
        i = None 
    else:
        s, i = ezker

    while i != None:
        seg_ordenatuta.append(seg.pop(i))

        ezker = ezkerreko_gertuena(seg, s)

        if ezker == None:
            i = None 
        else:
            s, i = ezker

    s, i = eskuineko_gertuena(seg, s, img)

    while i != None:
        seg_ordenatuta.append(seg.pop(i))
        eskuin = eskuineko_gertuena(seg, s, img)
        if eskuin == None:
            i = None 
        else:
            s, i = eskuin
    
    return seg_ordenatuta

# Ordenatutako lista bat pasatuta, beharrezkoak diren puntuak bakarrik gordetzen ditu 
def kendu_beharrezkoak_ez_direnak(seg_ordenatuta):

    lista_garbia = []
    x = None

    for i in range(len(seg_ordenatuta)):
        a = i-1
        o = i+1
        if i==0:
            a = len(seg_ordenatuta)-1
        if i==len(seg_ordenatuta)-1:
            o = 0

        if abs(seg_ordenatuta[a][0]-seg_ordenatuta[o][0])<3 and ((seg_ordenatuta[i][1]<seg_ordenatuta[a][1] and seg_ordenatuta[i][1]>seg_ordenatuta[o][1]) or (seg_ordenatuta[i][1]>seg_ordenatuta[a][1] and seg_ordenatuta[i][1]<seg_ordenatuta[o][1])):
            if x == False:
                lista_garbia.append(seg_ordenatuta[i-1])
            x = True

            continue
        if abs(seg_ordenatuta[a][1]-seg_ordenatuta[o][1])<3 and ((seg_ordenatuta[i][0]<seg_ordenatuta[a][0] and seg_ordenatuta[i][0]>seg_ordenatuta[o][0]) or (seg_ordenatuta[i][0]>seg_ordenatuta[a][0] and seg_ordenatuta[i][0]<seg_ordenatuta[o][0])):
            if x == True:
                lista_garbia.append(seg_ordenatuta[i-1])
            x = False
            
            continue
        lista_garbia.append(seg_ordenatuta[i])
    return lista_garbia

#Maskara bat sartuta, bere ertzen koordenatuak itzultzen ditu. (Txurro bat da)
def get_seg(img):
    xhasi = []
    xbukatu = []
    t = 0 #Errenkadan maskara erregistratu den edo ez, t=0 ez, t=1 bai eta gorde da, t=2 bai baina ez da gorde
    aurrekoah = -1 #Hasierako ertz bat bilatutakoan, zutabearen indizea gordetzen da, zutabe berean duten bilatzen diren ertzak ez gordetzeko
    aurrekoab = -1 #Bukaerako ertz bat bilatutakoan, zutabearen indizea gordetzen da, zutabe berean duten bilatzen diren ertzak ez gordetzeko
    for i, m in enumerate(img): #i.errenkada eta m pixel zerrenda
        for j, g in enumerate(m): #j.zutabea eta g pixel zerrenda
            if g[0]>200 and t==0: #Txuria bada eta ez bada maskara erregistratu errenkadan
                if j != aurrekoah and (len(m)-1)!=j: #Ez bada aurreko zutabe bera
                    xhasi.append((j, i)) #Gorde maskara hasi den pixelen koordenatua
                    aurrekoah = j #Eguneratu zutabea
                    t = 1 #Erregistratu maskara
                    # continue
                elif (img.shape[0]-1)==i: #Argazkiko azken errenkada bada
                    xhasi.append((j, i)) #Gorde maskara hasi den pixelen koordenatua
                    t = 1 #Erregistratu maskara
                    # continue
                elif (len(m)-1)==j and img[i+1][j][0]<200: #Azken zutabea bada eta behekoa beltza bada
                    xhasi.append((j, i)) #Gorde maskara hasi den pixelen koordenatua
                    t = 0
                    # continue
                elif (img[i+1][j][0]<200 and img[i+1][j+1][0]<200): #Maskararen azken errenkada bada (ezker beheko ertza, lerro zuzenetan)
                    xhasi.append((j, i)) #Gorde maskara hasi den pixelen koordenatua
                    t = 1 #Erregistratu maskara
                    # continue
                else: #Maskararen ezkerreko ertz bat bada, baina dagoeneko zutabea gorde bada
                    t = 2 #Erregistratu maskara, baina ez dela gorde
                    # continue

            if (img.shape[0]-1)==i: #Argazkiko azken errenkada bada
                if g[0]>200 and (len(m)-1)==j: #Txuria bada eta azken zutabea bada
                    xbukatu.append((j, i)) #Gorde maskara bukatu den pixelen koordenatua
                    t = 0
                    continue
                elif g[0]<200 and (t==1 or t==2): #Beltza bada eta maskara erregistratu abda
                    xbukatu.append((j-1, i)) #Gorde maskara bukatu den pixelen koordenatua (aurreko zutabea)
                    t = 0 #Maskara bukatu dela
                    continue

            elif ((t==1 or t==2)): #Tableroa erregistratu bada
                if g[0]<200: #Beltza bada
                    if j-1 != aurrekoab: #Ez bada aurreko zutabe bera
                        xbukatu.append((j-1, i)) #Gorde maskara bukatu den pixelen koordenatua (aurreko zutabea)
                        t = 0 #Maskara bukatu dela
                        aurrekoab = j-1 #Eguneratu zutabea (aurreko zutabea)
                        continue
                    else: #Aurreko zutabe bera bada
                        t = 0 #Maskara bukatu dela baina ez dela gorde
                        continue

                elif img[i+1][j][0]<200 and img[i+1][j-1][0]<200 and j == aurrekoab: #Maskararen azken errenkada bada eta aurreko zutabe bera bada (eskuin beheko ertza, lerro zuzenetan)
                    xbukatu.append((j, i)) #Gorde maskara bukatu den pixelen koordenatua
                    t = 0 #Maskara bukatu dela
                    continue

            if g[0]>200 and (len(m)-1)==j: #Txuria bada, eta azken zutabean bada
                t = 0 #Maskara bukatu dela
                if j != aurrekoab: #Ez bada aurreko zutabe bera
                    xbukatu.append((j, i)) #Gorde maskara bukatu den pixelen koordenatua
                    aurrekoab = j #Eguneratu zutabea

    seg = (xhasi + list(set(xbukatu) - set(xhasi))) #Gehitu bi listak

    seg_ordenatuta = ordenatu_lista(seg, img) #Ordenatu

    lista_garbia = kendu_beharrezkoak_ez_direnak(seg_ordenatuta) #Kendu behar ez diren puntuak

    seg_ordenatuta = ordenatu_lista(lista_garbia, img) #Ordenatu

    lista_garbia = kendu_beharrezkoak_ez_direnak(seg_ordenatuta) #Kendu behar ez diren puntuak

    return np.array(lista_garbia).flatten().tolist()
    
def biggestContour(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if area > max_area:
                biggest = approx
                max_area = area 
                bbox = cv2.boundingRect(i)
    return biggest.flatten().tolist(), bbox, max_area 


def get_annotations(id, name, image):

    # creating a dictionary to store the image and its annotations
    im_dict = {}
    im_dict['id'] = id
    im_dict['file_name'] = name
    im_dict['image'] = image
    im_dict['width'] = image.shape[1]
    im_dict['height'] = image.shape[0]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    seg, bbox, area = biggestContour(image)
    if area==0:
        coco_data = {
                
                'images': [
                    {
                        'id': im_dict['id'],
                        'width': im_dict['width'],
                        'height': im_dict['height'],
                        'file_name': im_dict['file_name']
                    }
                ],
                'annotations': [{
                    'id': im_dict['id'],
                    'iscrowd': 0,
                    'image_id': im_dict['id'],
                    'category_id': 0,
                    'segmentation': [[]],
                    'bbox': [],
                    'area': 0
                }],
                'categories': []
            }
        
    else:
        coco_data = {
            
            'images': [
                {
                    'id': im_dict['id'],
                    'width': im_dict['width'],
                    'height': im_dict['height'],
                    'file_name': im_dict['file_name']
                }
            ],
            'annotations': [],
            'categories': []
        }

        # looping through the contours and adding them to the dictionary
        # for contour in im_dict['contours']:
        #     contour = np.array(contour, dtype=np.float32)

        #     # checking if the contour has enough points
        #     if contour.shape[0] < 3:
        #         continue

        # adding the contour to the dictionary
        coco_data['annotations'].append({
            'id': im_dict['id'],
            'iscrowd': 0,
            'image_id': im_dict['id'],
            'category_id': 0,
            'segmentation': [seg],
            'bbox': bbox,
            'area': area
        })

    return coco_data


def main():
    args = parse_args()

    dir_path = args.images
    img_path = dir_path + '/images/'
    mask_path = dir_path + '/masks/'
    dest_path = args.dest
    size = int(args.size)
    overlap = args.overlap

    dest_ann_path = dest_path + '/annotations/'
    dest_images_path = dest_path + '/images/'
    dest_masks_path = dest_path + '/masks/'


    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    if not os.path.exists(dest_ann_path):
        os.makedirs(dest_ann_path)
    if not os.path.exists(dest_images_path):
        os.makedirs(dest_images_path)
    if not os.path.exists(dest_masks_path):
        os.makedirs(dest_masks_path)
    

    arr = os.listdir(img_path)

    images = []
    annotations = []

    categories = [{'id': 0,'name': 'tableroa'}]

    image_id = 0

    for id, img in enumerate(tqdm(arr, position=0)):
        large_image = cv2.imread(img_path+img)
        large_mask = cv2.imread(mask_path+img)
        shape = large_image.shape
        if shape[0]<shape[1]:
            large_image = large_image[:shape[1]-shape[0],:]
        elif shape[0]>shape[1]:
            large_image = large_image[:,:shape[0]-shape[1]]

        patchk = shape[0]//size
        if overlap == None:
            patches = patchify(large_image, (size, size, 3), size)
        else:
            patches = patchify(large_image, (size, size, 3), int(overlap))
        patches_mask = patchify(large_mask, (size, size, 3), size)
        patches = patches[:,:,0,...]
        patches_mask = patches_mask[:,:,0,...]

        for i in range(len(patches[0])): 
            for j in range(len(patches[1])):
                number = '{0:04}'.format(id)
                number2 = '{0:04}'.format(int(patchk*i + j))
                name = 'image_{}_{}.jpg'.format(number, number2)
                filename_img = dest_images_path + name
                filename_mask = dest_masks_path + name

                cv2.imwrite(filename_img, patches[i, j], [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(filename_mask, patches_mask[i, j], [cv2.IMWRITE_PNG_COMPRESSION, 0])
                ann = get_annotations(image_id, name, patches_mask[i,j])
                images.append(ann['images'][0])
                annotations.append(ann['annotations'][0])
                image_id += 1

    dataset = {'info': {"description":""}, 'images': images, 'annotations': annotations, "categories":categories}


    with open(dest_ann_path + 'coco.json', "w") as outfile: 
        json.dump(dataset, outfile, indent=3)

if __name__ == "__main__":
    main()