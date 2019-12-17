"""
  Voxnet implementation using pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxNet(nn.Module):
    def __init__(self,
    in_channels: int = 3,
    feat_size: int = 128,
    num_classes: int = 11,
    dropout: float = 0.,
    classifier_layer_dims: Iterable[int] = [128],
    feat_layer_dims: Iterable[int] = [14, 12, 6],
    activation=F.relu,
    batchnorm: bool = True,
    transposed_input: bool = False):

    super(VoxNet, self).__init__()

    if not isinstance(num_classes, int):
        raise TypeError('Argument num_classes must be of type int. '
                        'Got {0} instead.'.format(type(num_classes)))
    if not isinstance(dropout, float):
        raise TypeError('Argument dropout must be of type float. '
                        'Got {0} instead.'.format(type(dropout)))
    if dropout < 0 or dropout > 1:
        raise ValueError('Dropout ratio must always be in the range'
                         '[0, 1]. Got {0} instead.'.format(dropout))
    if not hasattr(classifier_layer_dims, '__iter__'):
        raise TypeError('Argument classifier_layer_dims is not iterable.')
    for idx, layer_dim in enumerate(classifier_layer_dims):
        if not isinstance(layer_dim, int):
            raise TypeError('Expected classifier_layer_dims to contain '
                            'int. Found type {0} at index {1}.'.format(
                                type(layer_dim), idx))

    # Add feat_size to the head of classifier_layer_dims (the output of
    # the PointNet feature extractor has number of elements equal to
    # has number of channels equal to `in_channels`).
    if not isinstance(classifier_layer_dims, list):
        classifier_layer_dims = list(classifier_layer_dims)
    classifier_layer_dims.insert(0, feat_size)

    # Note that `global_feat` MUST be set to True, for global
    # classification tasks.
    self.feature_extractor = PointNetFeatureExtractor(
        in_channels=in_channels, feat_size=feat_size,
        layer_dims=feat_layer_dims, global_feat=True,
        activation=activation, batchnorm=batchnorm,
        transposed_input=transposed_input
    )

    self.linear_layers = nn.ModuleList()
    if batchnorm:
        self.bn_layers = nn.ModuleList()
    for idx in range(len(classifier_layer_dims) - 1):
        self.linear_layers.append(nn.Linear(classifier_layer_dims[idx],
                                            classifier_layer_dims[idx + 1]))
        if batchnorm:
            self.bn_layers.append(nn.BatchNorm1d(
                classifier_layer_dims[idx + 1]))

    self.last_linear_layer = nn.Linear(classifier_layer_dims[-1],
                                       num_classes)

    # Store activation as a class attribute
    self.activation = activation

    # Dropout layer (if dropout ratio is in the interval (0, 1]).
    if dropout > 0:
        self.dropout = nn.Dropout(p=dropout)

    else:
        self.dropout = None

    # Store whether or not to use batchnorm as a class attribute
    self.batchnorm = batchnorm

    self.transposed_input = transposed_input

    def forward(self, x):
    r"""Forward pass through the PointNet classifier.
    Args:
        x (torch.Tensor): Tensor representing a pointcloud
            (shape: :math:`B \times N \times D`, where :math:`B`
            is the batchsize, :math:`N` is the number of points
            in the pointcloud, and :math:`D` is the dimensionality
            of each point in the pointcloud).
            If self.transposed_input is True, then the shape is
            :math:`B \times D \times N`.
    """
    if not self.transposed_input:
        x = x.transpose(1, 2)

    # Number of points
    num_points = x.shape[2]
    if self.batchnorm:
        x = self.activation(self.bn_layers[0](self.conv_layers[0](x)))
    else:
        x = self.activation(self.conv_layers[0](x))

    # Pass through the remaining layers (until the penultimate layer).
    for idx in range(1, len(self.conv_layers) - 1):
        if self.batchnorm:
            x = self.activation(self.bn_layers[idx](
                self.conv_layers[idx](x)))
        else:
            x = self.activation(self.conv_layers[idx](x))

    # For the last layer, do not apply nonlinearity.
    if self.batchnorm:
        x = self.bn_layers[-1](self.conv_layers[-1](x))
    else:
        x = self.conv_layers[-1](x)

    # Max pooling.
    x = torch.max(x, 2, keepdim=True)[0]
    x = x.view(-1, self.feat_size)

    for idx in range(len(self.linear_layers) - 1):
        if self.batchnorm:
            x = self.activation(self.bn_layers[idx](
                self.linear_layers[idx](x)))
        else:
            x = self.activation(self.linear_layers[idx](x))
    # For penultimate linear layer, apply dropout before batchnorm
    if self.dropout:
        if self.batchnorm:
            x = self.activation(self.bn_layers[-1](self.dropout(
                self.linear_layers[-1](x))))
        else:
            x = self.activation(self.dropout(self.linear_layers[-1](x)))
    else:
        if self.batchnorm:
            x = self.activation(self.bn_layers[-1](
                self.linear_layers[-1](x)))
        else:
            x = self.activation(self.linear_layers[-1](x))
    # TODO: Use dropout before batchnorm of penultimate linear layer
    x = self.last_linear_layer(x)
    # return F.log_softmax(x, dim=1)
    return x



        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            # index of the corresponding row of the logits tensor with the highest raw value
            'pred_cls': tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the `logging_hook`.
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Define loss function (for both TRAIN and EVAL modes)
        # one-hot encoding
        #   indices: the locations of 1 values in the tensor.
        #   depth: the depth of the one-hot tensor, i.e., the number of target classes.
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=self.num_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        tf.summary.scalar("loss", loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['pred_cls'])}

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
