import os

import scipy

import util
import numpy as np
from scipy.spatial import distance
from numba import jit
import timeit
from math import log, e

lastLayer = np.asarray([0])
valid_layer_names = []

def extractSingleLayer(key, layer, reset=False, layerActivationReduction='jsdiverge', model=None, inData=None, y_inData=None):
    # calculates jensen shannon divergence/distance per layer
    # get js divergence between layers
    global lastLayer, valid_layer_names

    if reset:
        lastLayer = np.asarray([0])
        valid_layer_names = get_valid_layer_names(model)

    result = np.asarray([0])
    if key in valid_layer_names:
        if len(lastLayer) == 1 and lastLayer[0] == 0:
            # store the layer for future use
            lastLayer = layer
        else:
            starttime = timeit.default_timer()
            result, lastLayer = calculate(lastLayer, layer)

    return result

def get_valid_layer_names(model):
    dnn = util.getParameter('Dnn')
    # get valid layer data
    full_filename = os.path.join(os.getcwd(), 'modules_reduce', 'data',
                                 'jsdiverge_%s' % (dnn)) + '.csv'

    if not os.path.isfile(full_filename):
        # create file
        # get the dnn layer numbers
        layer_csv_lines = []
        layer_csv_lines.append('layer_name,layer_block\n')
        layer_names = model.get_layer_names()
        for l in layer_names:
            if 'mobilenet' in dnn:
                if 'conv2d' in l.lower():
                    layer_csv_lines.append('%s,valid\n' % (l))
                else:
                    layer_csv_lines.append('%s,%s\n' % (l, 'none'))
            else:
                layer_csv_lines.append('%s,%s\n' % (l, 'valid'))

        with open(full_filename, 'w') as f:
            f.writelines(layer_csv_lines)
        util.thisLogger.logInfo('jsdiverge valid layer file created in %s. Check that it is correct.' % (full_filename))

    # read the file and get valid layer names
    valid_layer_names = []
    if os.path.isfile(full_filename):
        with open(full_filename, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            l = line.replace('\n', '').split(',')
            if l[1] != 'none':
                valid_layer_names.append(l[0])

    return valid_layer_names

@jit(nopython=False)
def pad(key, layer):
    padLength = np.asarray(util.getParameter('PadLength.%s'%(util.getParameter('Dnn'))))
    if padLength > layer.shape[1]:
        # pad with zero's to make them all the same size
        paddedArray = np.zeros((layer.shape[0], padLength))
        paddedArray[:layer.shape[0], :layer.shape[1]] = layer
    elif padLength == layer.shape[1]:
        paddedArray = layer
    else:
        util.thisLogger.logInfo(
            'Could not apply padding to layer %s, as data length of %s exceeds padding length of %s' % (
            key, layer.shape[1], padLength))

    return paddedArray

# @jit(nopython=False)
def entropy(arry, base=None):
# def entropy(labels):
  """ Computes entropy of label distribution. """

  ents = np.array(len(arry))
  for i, labels in enumerate(arry):
      n_labels = len(labels)

      if n_labels <= 1:
        return 0

      value,counts = np.unique(labels, return_counts=True)
      probs = counts / n_labels
      n_classes = np.count_nonzero(probs)

      if n_classes <= 1:
        return 0

      ent = 0.

      # Compute entropy
      base = e if base is None else base
      # base = e
      for i in probs:
        ent -= i * log(i, base)
      ents[i] = ent

  return ents

@jit(nopython=False)
def rel_entr(x, y):
    if np.isnan(x) or np.isnan(y):
        return np.nan
    if x > 0 and y > 0:
        return x * log(x / y)
    elif x == 0 and y >= 0:
        return 0
    else:
        return np.inf

@jit(nopython=False)
def rel_entr_array(x_arry, y_arry):
    ents = np.zeros(x_arry.shape)
    for i, arry in enumerate(x_arry):
        for j, (x, y) in enumerate(zip(x_arry[i], y_arry[i])):
            ent = rel_entr(x, y)
            ents[i][j] = ent
    return ents


@jit(nopython=False)
def calculate(lastLayer, layer): # here the lastlayer means the previous layer to the current one
    value1 = lastLayer
    value2 = layer

    value1_abs = np.absolute(value1)
    value2_abs = np.absolute(value2)

    p = value1_abs
    q = value2_abs
    axis=1
    keepdims = True

    q = np.asarray(q)
    p = p / np.sum(p, axis=axis, keepdims=True)
    q = q / np.sum(q, axis=axis, keepdims=True)
    m = (p + q) / 2.0
    left = rel_entr_array(p,m)
    right = rel_entr_array(q, m)

    left_sum = np.sum(left, axis=axis, keepdims=keepdims)
    right_sum = np.sum(right, axis=axis, keepdims=keepdims)
    js = left_sum + right_sum

    return np.sqrt(js / 2.0), value2


def extractMultiLayer(actDict, layerActivationReduction='jsdiverge', reset=False):
    return actDict