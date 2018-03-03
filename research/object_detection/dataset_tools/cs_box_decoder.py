# -*- coding: utf-8 -*-
"""
Created on Sat Mar 03 13:09:12 2018

This is a helper function which can generate bounding boxes from cityscapes instance annotations.
Change line 16 if this file is not run from the helpers folder of the cityscapes devkit.
Uncomment lines 20 and 61-73 for a usage example and visualization

@author: Oeljeklaus
"""

import sys
import os

#sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..') ) )
sys.path.append( '/media/malte/samba_cityscapes/scripts' )

from helpers.labels import *

import PIL.Image as Image
#import PIL.ImageDraw as ImageDraw

import numpy as np

def get_gtBoxes(input_fn, class_id, num_classes):
  
  head, tail = os.path.split(input_fn)
  head = head.replace('leftImg8bit','gtFine')
  tail = tail.replace('leftImg8bit','gtFine_instanceTrainIds')
  seg_gt = os.path.join(head, tail)

  # credits to https://github.com/TuSimple/mx-maskrcnn/blob/master/rcnn/dataset/cityscape.py
  assert os.path.exists(seg_gt), 'Path does not exist: {}, did you run \"createTrainIdInstanceImgs.py\"?'.format(seg_gt)
  im = Image.open(seg_gt)
  pixel = list(im.getdata())
  pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
  boxes = []
  gt_classes = []
  ins_id = []
  gt_overlaps = []
  for c in range(0, len(class_id)):
    px = np.where((pixel >= class_id[c] * 1000) & (pixel < (class_id[c] + 1) * 1000))
    if len(px[0]) == 0:
      continue
    ids = np.unique(pixel[px])
    for id in ids:
      px = np.where(pixel == id)
      x_min = np.min(px[1])
      y_min = np.min(px[0])
      x_max = np.max(px[1])
      y_max = np.max(px[0])
      if x_max - x_min <= 1 or y_max - y_min <= 1:
        continue
      boxes.append([x_min, y_min, x_max, y_max])
#      gt_classes.append(c)
      gt_classes.append(class_id[c])
      ins_id.append(id % 1000)
      overlaps = np.zeros(num_classes)
      overlaps[c] = 1
      gt_overlaps.append(overlaps)
  return np.asarray(boxes), np.asarray(gt_classes), np.asarray(ins_id), seg_gt, np.asarray(gt_overlaps)

#def main(argv):
#  testfile = 'G:/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'  
#  classes = ['car']
#  boxes, gt_classes, ins_id, seg_gt, gt_overlaps = get_gtBoxes(testfile, [l.trainId for l in labels if l.name in classes], len(classes))
#  im = Image.open(testfile)
#  draw = ImageDraw.Draw(im)
#  for i in range(0,boxes.shape[0]):
#    draw.rectangle(boxes[i,:].tolist(),fill=None,outline=(255,0,0))
#  im.show()  
#  return
#    
#if __name__=="__main__":
#  main(sys.argv)