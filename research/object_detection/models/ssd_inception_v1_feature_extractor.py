"""SSDFeatureExtractor for InceptionV1 features."""
import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from nets import inception_v1

slim = tf.contrib.slim


class SSDInceptionV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using InceptionV1 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None):
    """InceptionV1 Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
    """
    super(SSDInceptionV1FeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable, reuse_weights)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs.get_shape().assert_has_rank(4)
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    feature_map_layout = {
        'from_layer': ['Mixed_4c', 'Mixed_5c', '', '', '', ''],
        'layer_depth': [-1, -1, 512, 256, 256, 128],
    }

    with tf.control_dependencies([shape_assert]):
      with slim.arg_scope(self._conv_hyperparams):
        with tf.variable_scope('InceptionV1',
                               reuse=self._reuse_weights) as scope:
          _, image_features = inception_v1.inception_v1_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              final_endpoint='Mixed_5c',
#              min_depth=self._min_depth,
#              depth_multiplier=self._depth_multiplier,
              scope=scope)
          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)

    return feature_maps.values()
