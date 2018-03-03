# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert raw cityscapes detection dataset to TFRecord for object_detection.

Converts cityscapes detection dataset to TFRecords with a standard format allowing
  to use this dataset to train object detectors. 
  Permission can be requested at the main website.

  cityscapes detection dataset contains ??? training images. Using this code with
  the default settings will set aside the first ??? images as a validation set.
  This can be altered using the flags, see details below.

Example usage:
    python object_detection/dataset_tools/create_cityscapes_tf_record.py \
        --data_dir=/home/user/cityscapes \
        --output_path=/home/user/cityscapes.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf
import json

from random import shuffle
from lxml import etree
from skimage.draw import polygon

import scipy.misc

import sys
sys.path.insert(0, './')
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou

sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ )) ) )
from cs_box_decoder import *

tf.app.flags.DEFINE_string('data_dir', '', 'Location of root directory for the '
                           'data. Folder structure is assumed to be:'
                           '<data_dir>/training/label_2 (annotations) and'
                           '<data_dir>/data_object_image_2/training/image_2'
                           '(images).')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_path>_train.tfrecord.'
                           'And the TFRecord with the validation set will be'
                           'located at: <output_path>_val.tfrecord')
#tf.app.flags.DEFINE_list('classes_to_use', ['car', 'pedestrian', 'dontcare'],
#                         'Which classes of bounding boxes to use. Adding the'
#                         'dontcare class will remove all bboxs in the dontcare'
#                         'regions.')
tf.app.flags.DEFINE_string('label_map_path', 'data/cityscapes_label_map.pbtxt',
                           'Path to label map proto.')
tf.app.flags.DEFINE_integer('validation_set_size', '500', 'Number of images to'
                            'be used as a validation set.')
FLAGS = tf.app.flags.FLAGS


def convert_cityscapes_to_tfrecords(data_dir, output_path, classes_to_use,
                               label_map_path, validation_set_size):
  """Convert the cityscapes detection dataset to TFRecords.

  Args:
    data_dir: The full path to the unzipped folder containing the unzipped data
    output_path: The path to which TFRecord files will be written. The TFRecord
      with the training set will be located at: <output_path>_train.tfrecord
      And the TFRecord with the validation set will be located at:
      <output_path>_val.tfrecord
    classes_to_use: List of strings naming the classes for which data should be
      converted. Use the same names as presented in the cityscapes README file.
    label_map_path: Path to label map proto
    validation_set_size: How many images should be left as the validation set.
      (Ffirst `validation_set_size` examples are selected to be in the
      validation set).
  """
  label_map_dict = label_map_util.get_label_map_dict(label_map_path)
  train_count = 0
  val_count = 0
  
  input_files = []

  image_dir = os.path.join(data_dir,'leftImg8bit','train')
#  cities = os.walk(image_dir)
#  print(cities)
  for _, cities, _ in os.walk(image_dir):
    for d in cities:
      for _, _, fn in os.walk(os.path.join(image_dir, d)):
        input_files.extend([os.path.join(image_dir, d, f) for f in fn])
#        input_files.append(os.path.join(image_dir,d,f))
  
#  print(input_files)
  print('found ' + str(len(input_files)) + ' input images.')

#  road_dir = '/media/malte/samba_share/KITTI/data_road/training/image_2'

  train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
                                             output_path)
  val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
                                           output_path)

#  images = sorted(tf.gfile.ListDirectory(image_dir_Manuel))
  
#  print([os.path.join(seg_annotation_dir_Manuel,x[0:len(x)-4]+'__labels.json') for x in images])
  
#  images = [x for x in images if tf.gfile.Exists(os.path.join(seg_annotation_dir_Manuel,x[0:len(x)-4]+'__labels.json'))]


#os.path.join(dirname, '../GT_Segmentation_Annotations', filename[0:len(filename)-4]+'.json')


#  images = images + sorted(tf.gfile.ListDirectory(image_dir_InVerSiV_ED))
#  images = sorted(tf.gfile.ListDirectory(image_dir_InVerSiV_ED))
#  images = images + sorted(tf.gfile.ListDirectory('/media/malte/samba_share/KITTI/data_road/training/image_2'))
  
  shuffle(input_files)
  
  for img_name in input_files:
    print(img_name)
    
    is_validation_img = False

    img_anno = read_annotation_file(img_name, classes_to_use)


    # Filter all bounding boxes of this frame that are of a legal class, and
    # don't overlap with a dontcare region.
    # TODO(talremez) filter out targets that are truncated or heavily occluded.
    annotation_for_image = filter_annotations(img_anno, classes_to_use)

    example = prepare_example(img_name, annotation_for_image, label_map_dict)

    if is_validation_img:
      val_writer.write(example.SerializeToString())
      val_count += 1
    else:
      train_writer.write(example.SerializeToString())
      train_count += 1

  train_writer.close()
  val_writer.close()


def prepare_example(image_path, annotations, label_map_dict):
  """Converts a dictionary with annotations for an image to tf.Example proto.

  Args:
    image_path: The complete path to image.
    annotations: A dictionary representing the annotation of a single object
      that appears in the image.
    label_map_dict: A map from string label names to integer ids.

  Returns:
    example: The converted tf.Example.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)
  image = np.asarray(image)

  key = hashlib.sha256(encoded_png).hexdigest()

  big_width = int(image.shape[1])
  big_height = int(image.shape[0])

  image = pil.open(encoded_png_io)
  image = image.resize(size=(965,483),resample=pil.BILINEAR)
  image = np.asarray(image)

  width = int(image.shape[1])
  height = int(image.shape[0])


#  out = scipy.misc.toimage(image)#,high=255,low=0,mode='P')
#  out.save(image_path+'_small.png')

  key = hashlib.sha256(encoded_png).hexdigest()

#  width = int(image.shape[1])
#  height = int(image.shape[0])



  present_label_indicator = 0
  
#  print(len(annotations))
#  print(present_label_indicator)
  
  if len(annotations)>0:
    present_label_indicator = 1
    xmin_norm = annotations['2d_bbox_left'] / float(big_width)
    ymin_norm = annotations['2d_bbox_top'] / float(big_height)
    xmax_norm = annotations['2d_bbox_right'] / float(big_width)
    ymax_norm = annotations['2d_bbox_bottom'] / float(big_height)

    difficult_obj = [0]*len(xmin_norm)
  else:
    xmin_norm = []
    ymin_norm = []
    xmax_norm = []
    ymax_norm = []
    annotations['type']=''

    annotations['truncated']=[]
    annotations['alpha']=[]
    annotations['3d_bbox_height']=[]
    annotations['3d_bbox_width']=[]
    annotations['3d_bbox_length']=[]
    annotations['3d_bbox_x']=[]
    annotations['3d_bbox_y']=[]
    annotations['3d_bbox_z']=[]
    annotations['3d_bbox_rot_y']=[]
    
    difficult_obj = []
  

#  numpy_segmentation_map = np.ones([height, width], dtype=np.float)*255.0 # default to ignore label

  head, tail = os.path.split(image_path)
  head = head.replace('leftImg8bit','gtFine')
  tail = tail.replace('leftImg8bit','gtFine_labelIds')
  segmentation_label_filename = os.path.join(head, tail)
  
  if not tf.gfile.Exists(segmentation_label_filename):
    sys.exit('file ' + segmentation_label_filename + ' not found!')

  seg = pil.open(segmentation_label_filename)
  pixel = list(seg.getdata())
  pixel = np.array(pixel).reshape([seg.size[1], seg.size[0]])

#  for p in [x for x in data if x['label_class']=='person']:
#    rr, cc = polygon([v['y'] for v in p['vertices']], [v['x'] for v in p['vertices']], numpy_segmentation_map.shape)
#    numpy_segmentation_map[rr,cc]=3

  numpy_segmentation_map = np.zeros([big_height, big_width], dtype=np.float) # default to background label = 0
  
#  print(label_map_dict)
#  print(len(label_map_dict))
  
  for ld in label_map_dict:
    labelId = [l.id for l in labels if l.name==ld][0]
    numpy_segmentation_map[pixel==labelId]=label_map_dict[ld]

  labelId = [l.id for l in labels if l.name=='road'][0]
  numpy_segmentation_map[pixel==labelId]=len(label_map_dict)+1


#    print(labelId)
#    print(ld)
#    print(labels[:].id)
  
  
  numpy_segmentation_map = scipy.misc.imresize(numpy_segmentation_map, (height, width),interp='nearest')
#  out = scipy.misc.toimage(numpy_segmentation_map,high=255,low=0,mode='I')
#  out.save(segmentation_label_filename+'_reduced_dual.png')

  seg_key = hashlib.sha256(numpy_segmentation_map).hexdigest()
  present_label_indicator += 2

  if (len(annotations['type']) != len(xmin_norm)) or (len(xmax_norm) != len(ymin_norm)) or (len(ymax_norm) != len(ymin_norm)):
    print('len(annotations[\'type\'])='+str(len(annotations['type'])))
    print('len(xmin_norm)='+str(len(xmin_norm)))
    print('len(xmax_norm)='+str(len(xmax_norm)))
    print('len(ymin_norm)='+str(len(ymin_norm)))
    print('len(ymax_norm)='+str(len(ymax_norm)))
    print(((annotations['type'])))
    print(xmin_norm)
    print(xmax_norm)
    print(ymin_norm)
    print(ymax_norm)
    assert((len(annotations['type']) == len(xmin_norm)) and (len(xmax_norm) == len(ymin_norm)) and (len(ymax_norm) == len(ymin_norm)))
    

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),

      'present_label_indicator': dataset_util.int64_feature(present_label_indicator), # 0 -> unknown, 1 -> input + bb, 2 -> input + seg, 3 -> input + bb + seg

      'image/seg/height': dataset_util.int64_feature(height),
      'image/seg/width': dataset_util.int64_feature(width),
#      'image/seg/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
#      'image/seg/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/seg/key/sha256': dataset_util.bytes_feature(seg_key.encode('utf8')),
#      'image/seg/numpy_segmentation_map': dataset_util.bytes_feature(numpy_segmentation_map.tobytes()),
      'image/seg/numpy_segmentation_map': dataset_util.float_list_feature(np.reshape(numpy_segmentation_map,[-1]).tolist()),
#      'image/seg/numpy_segmentation_map': dataset_util.int64_feature(tf.convert_to_tensor(numpy_segmentation_map, dtype=tf.int64,name='segmentation_map_gt')),
#      'image/seg/numpy_segmentation_map': dataset_util.int64_feature(numpy_segmentation_map),
      'image/seg/format': dataset_util.bytes_feature('numpy'.encode('utf8')),

      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(
          [x.encode('utf8') for x in annotations['type']]),
      'image/object/class/label': dataset_util.int64_list_feature(
          [label_map_dict[x] for x in annotations['type']]),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.float_list_feature(
          annotations['truncated']),
      'image/object/alpha': dataset_util.float_list_feature(
          annotations['alpha']),
      'image/object/3d_bbox/height': dataset_util.float_list_feature(
          annotations['3d_bbox_height']),
      'image/object/3d_bbox/width': dataset_util.float_list_feature(
          annotations['3d_bbox_width']),
      'image/object/3d_bbox/length': dataset_util.float_list_feature(
          annotations['3d_bbox_length']),
      'image/object/3d_bbox/x': dataset_util.float_list_feature(
          annotations['3d_bbox_x']),
      'image/object/3d_bbox/y': dataset_util.float_list_feature(
          annotations['3d_bbox_y']),
      'image/object/3d_bbox/z': dataset_util.float_list_feature(
          annotations['3d_bbox_z']),
      'image/object/3d_bbox/rot_y': dataset_util.float_list_feature(
          annotations['3d_bbox_rot_y']),
  }))

  return example


def filter_annotations(img_all_annotations, used_classes):
  """Filters out annotations from the unused classes and dontcare regions.

  Filters out the annotations that belong to classes we do now wish to use and
  (optionally) also removes all boxes that overlap with dontcare regions.

  Args:
    img_all_annotations: A list of annotation dictionaries. See documentation of
      read_annotation_file for more details about the format of the annotations.
    used_classes: A list of strings listing the classes we want to keep, if the
    list contains "dontcare", all bounding boxes with overlapping with dont
    care regions will also be filtered out.

  Returns:
    img_filtered_annotations: A list of annotation dictionaries that have passed
      the filtering.
  """

  img_filtered_annotations = {}
  if len(img_all_annotations)==0:
    return img_filtered_annotations

  # Filter the type of the objects.
  relevant_annotation_indices = [
      i for i, x in enumerate(img_all_annotations['type']) if x in used_classes
  ]
  
  for key in img_all_annotations.keys():
    img_filtered_annotations[key] = (
        img_all_annotations[key][relevant_annotation_indices])

  if 'dontcare' in used_classes:
    dont_care_indices = [i for i,
                         x in enumerate(img_filtered_annotations['type'])
                         if x == 'dontcare']

    # bounding box format [y_min, x_min, y_max, x_max]
    all_boxes = np.stack([img_filtered_annotations['2d_bbox_top'],
                          img_filtered_annotations['2d_bbox_left'],
                          img_filtered_annotations['2d_bbox_bottom'],
                          img_filtered_annotations['2d_bbox_right']],
                         axis=1)

    ious = iou(boxes1=all_boxes,
               boxes2=all_boxes[dont_care_indices])

    # Remove all bounding boxes that overlap with a dontcare region.
    if ious.size > 0:
      boxes_to_remove = np.amax(ious, axis=1) > 0.0
      for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_filtered_annotations[key][np.logical_not(boxes_to_remove)])

  return img_filtered_annotations

def read_annotation_file(filename, classes):
  """Reads a cityscapes annotation file.

  Converts a cityscapes annotation file into a dictionary containing all the
  relevant information.

  Args:
    filename: the path to the input image.

  Returns:
    anno: A dictionary with the converted annotation information. See annotation
    README file for details on the different fields.
  """
  
  boxes, gt_classes, _, _, _ = get_gtBoxes(filename, [l.trainId for l in labels if l.name in classes], len(classes))
  
  # box definition [x_min, y_min, x_max, y_max]
  
#  print([l.name for g in gt_classes for l in labels if l.trainId==g])
#  print([l.name for l in labels if l.trainId==g for g in gt_classes])
  
#  print([gt_classes[i] for i in range(0, len(gt_classes))])
#  print([classes[gt_classes[i]] for i in range(0, len(gt_classes))])
#  print([boxes[i,0] for i in range(0, len(gt_classes))])
#  print([boxes[i,1] for i in range(0, len(gt_classes))])
#  print([boxes[i,2] for i in range(0, len(gt_classes))])
#  print([boxes[i,3] for i in range(0, len(gt_classes))])

  anno = {}
  anno['type'] = np.array([l.name for g in gt_classes for l in labels if l.trainId==g])
  anno['truncated'] = np.array([False for g in gt_classes])   # default to False
  anno['occluded'] = np.array([False for g in gt_classes])    # default to False
  anno['alpha'] = np.array([False for g in gt_classes])       # default to False

  anno['2d_bbox_left'] = np.array([boxes[i,0] for i in range(0, len(gt_classes))], dtype=np.float)
  anno['2d_bbox_top'] = np.array([boxes[i,1] for i in range(0, len(gt_classes))], dtype=np.float)
  anno['2d_bbox_right'] = np.array([boxes[i,2] for i in range(0, len(gt_classes))], dtype=np.float)
  anno['2d_bbox_bottom'] = np.array([boxes[i,3] for i in range(0, len(gt_classes))], dtype=np.float)

  anno['3d_bbox_height'] = np.array([0 for g in gt_classes], dtype=np.float)    # default to 0
  anno['3d_bbox_width'] = np.array([0 for g in gt_classes], dtype=np.float)     # default to 0
  anno['3d_bbox_length'] = np.array([0 for g in gt_classes], dtype=np.float)    # default to 0
  anno['3d_bbox_x'] = np.array([0 for g in gt_classes], dtype=np.float)         # default to 0
  anno['3d_bbox_y'] = np.array([0 for g in gt_classes], dtype=np.float)         # default to 0
  anno['3d_bbox_z'] = np.array([0 for g in gt_classes], dtype=np.float)         # default to 0
  anno['3d_bbox_rot_y'] = np.array([0 for g in gt_classes], dtype=np.float)     # default to 0

  return anno


#
#def read_bbox_annotation_file_Manuel(filename):
#  """Reads a bbox annotation file from Manuels dataset.
#
#  Converts a annotation file from Manuels dataset into a dictionary containing all the
#  relevant information. The dictionary format is the same as with KITTI.
#
#  Args:
#    filename: the path to the annotataion text file.
#
#  Returns:
#    anno: A dictionary with the converted annotation information. See annotation
#    README file for details on the different fields.
#  """
#  if not tf.gfile.Exists(filename):
#    return {}
#
#  with tf.gfile.GFile(filename, 'r') as fid:
#    xml_str = fid.read()
#  xml = etree.fromstring(xml_str)
#  data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
#
#  anno = {}
#  anno['type'] = np.array([x['name'].lower() for x in data['object']])
#  anno['truncated'] = np.array([False for x in data['object']])   # default to False
#  anno['occluded'] = np.array([False for x in data['object']])    # default to False
#  anno['alpha'] = np.array([False for x in data['object']])       # default to False
#
#  anno['2d_bbox_left'] = np.array([x['bndbox']['xmin'] for x in data['object']], dtype=np.float)
#  anno['2d_bbox_top'] = np.array([x['bndbox']['ymin'] for x in data['object']], dtype=np.float)
#  anno['2d_bbox_right'] = np.array([x['bndbox']['xmax'] for x in data['object']], dtype=np.float)
#  anno['2d_bbox_bottom'] = np.array([x['bndbox']['ymax'] for x in data['object']], dtype=np.float)
#
#  anno['3d_bbox_height'] = np.array([0 for x in data['object']], dtype=np.float)    # default to 0
#  anno['3d_bbox_width'] = np.array([0 for x in data['object']], dtype=np.float)     # default to 0
#  anno['3d_bbox_length'] = np.array([0 for x in data['object']], dtype=np.float)    # default to 0
#  anno['3d_bbox_x'] = np.array([0 for x in data['object']], dtype=np.float)         # default to 0
#  anno['3d_bbox_y'] = np.array([0 for x in data['object']], dtype=np.float)         # default to 0
#  anno['3d_bbox_z'] = np.array([0 for x in data['object']], dtype=np.float)         # default to 0
#  anno['3d_bbox_rot_y'] = np.array([0 for x in data['object']], dtype=np.float)     # default to 0
#
#  return anno
#
#def read_bbox_annotation_file_ED(filename):
#  """Reads a bbox annotation file from ED dataset.
#
#  Converts a annotation file from ED dataset into a dictionary containing all the
#  relevant information. The dictionary format is the same as with KITTI.
#
#  Args:
#    filename: the path to the annotataion text file.
#
#  Returns:
#    anno: A dictionary with the converted annotation information. See annotation
#    README file for details on the different fields.
#  """
#
##  print(filename)
#
#  if not tf.gfile.Exists(filename):
#    return {}
#
#  with tf.gfile.GFile(filename, 'r') as fid:
#    json_str = fid.read()
#  data = json.loads(json_str)
#  data = [x for x in data['labels'] if x['label_type']=='box'] # filter for bounding boxes
#  data = [x for x in data if x['label_class']!=None and x['size']['x']!=0 and x['size']['y']!=0] # filter for valid bounding boxes
#
##  print(data)
#
#  anno = {}
#  anno['type'] = np.array([x['label_class'].lower() for x in data])
#  anno['truncated'] = np.array([False for x in data])   # default to False
#  anno['occluded'] = np.array([False for x in data])    # default to False
#  anno['alpha'] = np.array([False for x in data])       # default to False
#
#  anno['2d_bbox_left'] = np.array([x['centre']['x']-x['size']['x']/2.0 for x in data], dtype=np.float)
#  anno['2d_bbox_top'] = np.array([x['centre']['y']-x['size']['y']/2.0 for x in data], dtype=np.float)
#  anno['2d_bbox_right'] = np.array([x['centre']['x']+x['size']['x']/2.0 for x in data], dtype=np.float)
#  anno['2d_bbox_bottom'] = np.array([x['centre']['y']+x['size']['y']/2.0 for x in data], dtype=np.float)
#
#  anno['3d_bbox_height'] = np.array([0 for x in data], dtype=np.float)    # default to 0
#  anno['3d_bbox_width'] = np.array([0 for x in data], dtype=np.float)     # default to 0
#  anno['3d_bbox_length'] = np.array([0 for x in data], dtype=np.float)    # default to 0
#  anno['3d_bbox_x'] = np.array([0 for x in data], dtype=np.float)         # default to 0
#  anno['3d_bbox_y'] = np.array([0 for x in data], dtype=np.float)         # default to 0
#  anno['3d_bbox_z'] = np.array([0 for x in data], dtype=np.float)         # default to 0
#  anno['3d_bbox_rot_y'] = np.array([0 for x in data], dtype=np.float)     # default to 0
#
##  print(anno)
#
##  print([x for x in data])
##  print([x in data['labels'] if x['label_type']=='box'])
##  print([x['label_type'] for x in data['labels']])
##  print([x for x in data['labels'] if x['label_type']=='box'])
#  return anno
    
    
#def read_annotation_file(filename):
#  """Reads a KITTI annotation file.
#
#  Converts a KITTI annotation file into a dictionary containing all the
#  relevant information.
#
#  Args:
#    filename: the path to the annotataion text file.
#
#  Returns:
#    anno: A dictionary with the converted annotation information. See annotation
#    README file for details on the different fields.
#  """
#  with open(filename) as f:
#    content = f.readlines()
#  content = [x.strip().split(' ') for x in content]
#
#  anno = {}
#  anno['type'] = np.array([x[0].lower() for x in content])
#  anno['truncated'] = np.array([float(x[1]) for x in content])
#  anno['occluded'] = np.array([int(x[2]) for x in content])
#  anno['alpha'] = np.array([float(x[3]) for x in content])
#
#  anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
#  anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
#  anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
#  anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
#
#  anno['3d_bbox_height'] = np.array([float(x[8]) for x in content])
#  anno['3d_bbox_width'] = np.array([float(x[9]) for x in content])
#  anno['3d_bbox_length'] = np.array([float(x[10]) for x in content])
#  anno['3d_bbox_x'] = np.array([float(x[11]) for x in content])
#  anno['3d_bbox_y'] = np.array([float(x[12]) for x in content])
#  anno['3d_bbox_z'] = np.array([float(x[13]) for x in content])
#  anno['3d_bbox_rot_y'] = np.array([float(x[14]) for x in content])
#
#  return anno


def main(_):
  convert_cityscapes_to_tfrecords(
      data_dir=FLAGS.data_dir,
      output_path=FLAGS.output_path,
      classes_to_use=['car', 'truck', 'bus', 'person', 'motorcycle', 'bicycle'], #FLAGS.classes_to_use,
      label_map_path=FLAGS.label_map_path,
      validation_set_size=FLAGS.validation_set_size)

if __name__ == '__main__':
  tf.app.run()
