import argparse
import subprocess


#conda activate coco
#cd C:\Users\a.sudupe\Desktop\TFG\Notebooks and Scripts\data_utils
#python setup.py -d C:\Users\a.sudupe\Desktop\TFG\Datasets\Proba_osoa\Hasierakoa -s 512 

def parse_args():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument("-d", "--data",help="path to data")
    parser.add_argument("-s", "--size",help="size of the patch")
    parser.add_argument("-n", "--donot", default= 0,help="from this part")
    
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    data_path = args.data
    size = args.size
    donot = int(args.donot)
    donot = range(donot, 5, 1)

    if 1 in donot:
        # print("python ann2mask.py {}".format('-d ' + data_path))
        print('####Creating masks:####')
        a = subprocess.run("python ann2mask.py" + ' -i ' + data_path, shell=True)
        if a.returncode !=0:
            print(a.returncode)
            return a.returncode
    
    dest_path = data_path + '/../patched'
    if 2 in donot:
        print('####Creating patches:####')
        a = subprocess.run("python patch_images.py" + ' -i ' + data_path + ' -d ' + dest_path + ' -s ' + size)
        if a.returncode !=0:
            print(a.returncode)
            return a.returncode
        
    data_path = dest_path
    ann_path = dest_path + '/annotations/coco.json'
    dest_path = dest_path + '/../patched_train_test'
    size = '0.5'
    if 3 in donot:
        print('####Spliting into (train, test, val):####')


        a = subprocess.run("python split_train_test.py" + ' -i ' + data_path + ' -a ' + ann_path + ' -d ' + dest_path + ' -s ' + size)
        if a.returncode !=0:
            print(a.returncode)
            return a.returncode
        
    if 4 in donot:
        print('####Create Yolo dataset:####')
        data_path = dest_path
        dest_path = dest_path + '/../patched_yolo'
        a = subprocess.run("python coco2yolo.py" + ' -i ' + data_path + ' -d ' + dest_path)
        if a.returncode !=0:
            print(a.returncode)
            return a.returncode
    

if __name__ == "__main__":
    main()