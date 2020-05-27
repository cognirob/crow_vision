# yolact inference
import sys 
import os
import torch
import cv2
import numpy as np
sys.path.append(os.path.relpath("../crow_vision_yolact"))
# sys.path.append(os.path.relpath("yolact"))
from yolact import Yolact
from data import set_cfg
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess
from eval import prep_display 
from data.config import Config

def find_objects(image, numobj):
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(image.astype(np.uint8),cv2.CV_32S)
    
    # visualize centroids
    for centroid in centroids[1:numobj + 1]:
        cv2.circle(image, (int(centroid[0]), int(centroid[1])), radius=10, color=(255, 0, 0),
                   thickness=-1)
        cv2.imshow('centroids',image)
        cv2.waitKey(1000)
    try:
        return centroids[1]
    except:
        return centroids
        
class YolactInfTool():

    def __init__(self, trained_model='./data/yolact/weights/weights_yolact_kuka_13/crow_base_59_400000.pth', top_k=15, score_threshold=0.15,):
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.model = trained_model

        global args
        args=Config({})
        args.top_k = top_k
        args.score_threshold = score_threshold
        # set here everything that would have been set by parsing arguments in yolact/eval.py:
        args.display_lincomb = False
        args.crop = False
        args.display_fps = False
        args.display_text = True
        args.display_bboxes = True
        args.display_masks =True
        args.display_scores = True

        # CUDA setup
        torch.backends.cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        set_cfg('crow_base_config')
        self.net = Yolact().cuda()
        self.net.load_weights(trained_model)
        self.net.eval()
        self.net.detect.use_fast_nms = True
        self.net.detect.use_cross_class_nms = False

    def raw_inference(self,image):
        #obtaining "raw" classes, scores, boxes, masks, centroids
        with torch.no_grad():
            frame = torch.from_numpy(image).cuda().float()
            h, w, _ = image.shape
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds_raw = self.net(batch)
            
            t = postprocess(preds_raw, w, h, score_threshold=args.score_threshold)
            idx = t[1].argsort(0, descending=True)[:args.top_k]
            classes, scores, boxes, masks = [x[idx].cpu().numpy() for x in t[:4]] #x[idx] or x[idx].cpu().numpy()
            centroids=[]
            for i in range(len(masks)):
                cv2.imshow('bin_mask',masks[i])
                cv2.waitKey(200)
                #if classes[i] > 0: #kuka class
                centroids.append(find_objects(masks[i],1)) #in pixel space
        return classes, scores, boxes, masks, centroids

    def visualized_inference(self,image):
        #obtaining one evaluated img with all masks, boxes, scores and class labels
        with torch.no_grad():
            frame = torch.from_numpy(image).cuda().float()
            h, w, _ = image.shape
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = self.net(batch)
            
            img_numpy = prep_display(preds, frame, None, None, undo_transform=False, class_color=True, mask_alpha=0.45, fps_str='', args=args)
            img_numpy = img_numpy[:, :, (2, 1, 0)]
            
            # cv2.imshow('img_yolact{}'.format(camera_id),img_numpy)
            # cv2.waitKey(1)
        return img_numpy
