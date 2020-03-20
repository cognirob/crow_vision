#%matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import cv2 as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (6.0, 8.0)

dataDir='../crow_vision_yolact/data/yolact/datasets'
datasetNr='0'
dataType='train'
annFile='{}/dataset_kuka_env_pybullet_{}/{}/annotations.json'.format(dataDir,datasetNr,dataType) #path to annotation.json file
num_img_to_display = 100

# initialize COCO api for instance annotations
coco=COCO(annFile)
# get all images containing given categories
catIds = 4 #for old dataset format (until dataset 13) use the id
catIds = coco.getCatIds(catNms=['wrench','wood_round']); #for new dataset format (since dataset 13) can use the name
imgIds = coco.getImgIds(catIds=catIds )
#imgIds = coco.getImgIds(imgIds = [100])

num=0
while num < num_img_to_display:
    # select one image for display at random
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

    # load (and display) image
    I = io.imread('%s/dataset_kuka_env_pybullet_%s/%s/%s'%(dataDir,datasetNr,dataType,img['file_name']))
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # load and display instance annotations
    plt.imshow(I); 
    plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.show()

    num += 1

