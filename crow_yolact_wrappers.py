## Hack wrappers for yolact for our usecases
import sys
import os
sys.path.append(os.path.abspath("./yolact"))
from eval import prep_display as yolact_prep_display #we wish to override prep_display signature
from data.config import Config

global args
args=Config({})
# set here everything that would have been set by parsing arguments in yolact/eval.py:
args.top_k = 7
args.display_lincomb = False
args.crop = False
args.score_threshold = 0.15
args.display_fps = False
args.display_text = True
args.display_bboxes = True
args.display_masks =True
args.display_scores = True

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str='', args=args):
  return yolact_prep_display(dets_out, img, h, w, undo_transform, class_color, mask_alpha, fps_str, args=args) #pass overriden args here
