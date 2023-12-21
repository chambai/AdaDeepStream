import util
import numpy as np
from modules_reduce import pad, jsdiverge, flatten


def extractSingleLayer(key, layer, reset=False, layerActivationReduction='jsdivergelast', model=None, inData=None, y_inData=None):
    # claculates jensen shannon divergence/distance between neighbouring layers and between the final hidden classification layer
    result = np.asarray([0])

    # get original jsdivergence calculation between consecutive layers
    jsResult = jsdiverge.extractSingleLayer(key, layer, reset, layerActivationReduction='jsdiverge', model=model, inData=inData, y_inData=y_inData)

    # get the activations for the final hidden layer and pad them to be the same size as the other layers
    if key in jsdiverge.valid_layer_names:
        lastHiddenLayer = getLayerActivations(model, inData)
        lastHiddenLayer = flatten.extractSingleLayer(key, lastHiddenLayer)
        lastHiddenLayer = pad.extractSingleLayer(key, lastHiddenLayer)

        # get jsdivergence calculation between this layer and the final hidden layer
        jsLast, _ = jsdiverge.calculate(layer, lastHiddenLayer) # here lastlayer means the last hidden layer of the network

        # concatenate the standard jsdivergence result with the jsdivergence between the layer and last layer
        if jsResult.size == 1 and jsResult[0] == 0:
            result = jsLast
        else:
            result = np.hstack((jsResult, jsLast))

    return result

def extractMultiLayer(actDict, layerActivationReduction='jsdiverge', reset=False):
    util.thisLogger.logInfo('NOT IMPLEMENTED')
    return actDict

def getLayerActivations(model, inData):
    layerNum = model.get_num_layers() - 2 # check this is the classification layer
    return model.get_layer_output(layerNum, inData)
