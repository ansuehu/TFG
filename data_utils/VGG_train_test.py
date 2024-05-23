import os
import json

dest_path = r"C:\Users\a.sudupe\Desktop\TFG\Datasets\Argazkiak_VGG"
anno_path_file = r"C:\Users\a.sudupe\Desktop\TFG\Datasets\backup\annotations_all\VGG.json"

# Define a list of image extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

imgs_list = {f:[filename for filename in os.listdir(os.path.join(dest_path, f)) if os.path.splitext(filename)[-1] in image_extensions] for f in os.listdir(dest_path)}

f = open(anno_path_file)
vgg = json.load(f)
f.close()

for t in ['train', 'val', 'test']:
    t_dict = {}
    imgs = imgs_list[t]
    for im in imgs:
        ann = vgg[im]
        t_dict[im]=ann

    out_file = open(dest_path+'/'+t+'/'+t+'.json', "w") 
  
    json.dump(t_dict, out_file, indent = 6) 
    
    out_file.close() 
