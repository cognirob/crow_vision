import cv2
import os
import glob

import skimage
import numpy as np

IN_DATASET="/tmp/dataset_kuka_env_pybullet_merge"
OUT_DATASET="/tmp/dataset_kuka_env_pybullet_merge_addnoise"

# read train/ + add noise + write 

n=0
for folder in ["/train/", "/test/"]:
    for i in glob.glob(IN_DATASET+folder+"*.jpg"): #TODO now you need to manually change for both *.png *.jpg
        f = os.path.basename(i)
        img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
        #add noise
        np.random.seed() #this is tricky, otherwise the distribution (with size=1) gives always the same number
        mod = np.random.randint(0, 3, 1, dtype=int)[0]
        n+=1
        print("at {} file {}".format(n,i))

        if mod == 1:
            amnt = np.random.uniform(0.0, 0.01)
            img_aug = skimage.util.random_noise(img, mode='gaussian', seed=42, var=amnt)
        elif mod == 2:
            amnt = np.random.uniform(0.0, 0.1)
            img_aug = skimage.util.random_noise(img, mode='s&p', seed=42, salt_vs_pepper=0.5, amount=amnt)
        else:
            amnt = np.random.uniform(0.0, 0.1)
            img_aug = skimage.util.random_noise(img, mode='speckle', seed=42, var=amnt)

        img_aug = cv2.convertScaleAbs(img_aug, alpha=(255.0)) #required after the skimage conversion, otherwise saved img is black
        #cv2.imshow("augmented img", img_aug); cv2.waitKey(100)
        cv2.imwrite(OUT_DATASET+folder+f, img_aug)

