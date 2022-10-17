# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:04:40 2019

@author: id127392
"""
import numpy as np
from collections import OrderedDict
import glob
from matplotlib import pyplot as plt
import util

lastHiddenLayer = np.asarray([0])
module = None


# ----------------------------------------------------------------------------
def loadModule(modName, dsData=None):
    global moduleName, module, results
    results = None
    moduleName = modName
    # module = __import__("das.analyse_" + moduleName, fromlist=[''])
    module = __import__(moduleName, fromlist=[''])
    if hasattr(module, 'setData'):
        module.setData(dsData)


# ----------------------------------------------------------------------------
def flatSetup(flatActivations, y_train):
    global module, moduleName
    util.thisLogger.logInfo("---------- start of %s activation extraction---------" % (moduleName))
    module.setupAnalysis(flatActivations, y_train)
    util.thisLogger.logInfo("----------- end of %s activaton extraction----------\n" % (moduleName))


# ---------------------------------------------------------------------------
def sliceDictOnKeys(dictionary, substring):
    return {k.lower(): v for k, v in dictionary.items() if substring.lower() in k}


# ---------------------------------------------------------------------------
def sliceDictNotOnKeys(dictionary, substring):
    return {k.lower(): v for k, v in dictionary.items() if substring.lower() not in k}

# ---------------------------------------------------------------------------
def getActivationData(model, inData, simActivations):
    if (simActivations == True):
        # simulate activations
        activations = {
            "activation1": np.array([[0.0], [0.1]]),
            "activation2": np.array([[0.0], [0.2]]),
            "activation3": np.array([[0.0], [0.3]]),
            "activation4": np.array([[0.0], [0.4]]),
            "activation5": np.array([[0.0], [0.5]]),
            "activation6": np.array([[0.0], [0.6]])
        }
        util.thisLogger.logInfo("*********Simulated activations*********")
    else:
        # get activations
        activations = getActivations(model, inData)

    # Set variables for the number of layers and the number of instances
    layerContainsName = util.getParameter('LayerContainsName')
    for name in layerContainsName:
        activations = sliceDictOnKeys(activations, name)

    layerDoesNotContainName = util.getParameter('LayerDoesNotContainName')
    for name in layerDoesNotContainName:
        activations = sliceDictNotOnKeys(activations, name)

    if (len(inData) == 1):
        displayImage = util.getParameter('DisplayImage')
        displayHeatmaps = util.getParameter('DisplayHeatmaps')
        actVisualisation = util.getParameter('DisplayActivationVis')
        if (displayImage == True):
            # display the original image
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.imshow(inData[0])

        if (displayHeatmaps == True):
            # display the heatmaps
            display_heatmaps(activations, inData[0], save=False)

        if (actVisualisation == True):
            # See here for colour map settings https://matplotlib.o
            # Display the activationsrg/3.1.0/tutorials/colors/colormaps.html
            display_activations(activations, cmap="hot", save=False)
    else:
        util.thisLogger.logInfo("Number of instances is > 1, not displaying activation visualisations")

    numLayers = np.asarray(list(activations.keys())).size

    # set all keys in the dictionary to lowercase
    activations = {k.lower(): v for k, v in activations.items()}

    # util.printDebug(activations)
    return activations, numLayers


# ---------------------------------------------------------------------------
def getActivationData2(model, inData, yData=None):
    activations = getActivations(model, inData, yData)
    return activations

# ----------------------------------------------------------------------------
def singleLayerActivationReduction(key, layer, reset=False, model=None, inData=None, y_inData=None):
    layerActivationReduction = util.getParameter('LayerActivationReduction')
    # if the layer activation reduction method is specified in teh test matrix, it will contain dots instead of commas.
    # separate out the dots into elements
    layerActRedArray = []
    for r in layerActivationReduction:
        if '.' in r:
            names = r.split('.')
            for n in names:
                layerActRedArray.append(n)
        else:
            layerActRedArray.append(r)

    for layerReductionName in layerActRedArray:
        util.thisLogger.logInfo('Applying layer activation reduction: %s' % (layerReductionName))
        if layerReductionName == 'none':
            layerReductionName = 'e_none'

        loadModule('modules_reduce.' + layerReductionName)
        layer = module.extractSingleLayer(key=key, layer=layer, reset=reset,
                                          layerActivationReduction=layerReductionName, model=model, inData=inData,
                                          y_inData=None)
    return np.asarray(layer)


# ----------------------------------------------------------------------------
def layerActivationReduction(actDict):
    layerActivationReduction = util.getParameter('LayerActivationReduction')
    for layerReductionName in layerActivationReduction:
        util.thisLogger.logInfo('Applying layer activation reduction: %s' % (layerReductionName))
        loadModule('modules_extract.' + layerReductionName)
        actDict = module.extractMultiLayer(actDict)
    return actDict

# -----------------------------------------------------------------------
def getActivations(model, inData, y_inData):
    global lastHiddenLayer
    lastLayer = np.asarray([0])
    actDict = OrderedDict()
    layerResults = np.zeros(1)
    batchNum = 0

    # ActivationLayerExtraction
    layerExtraction = 'single'
    # layers = util.getParameter("IncludedLayers")
    layers = 'all'

    if (layers == 'all'):
        layers = np.arange(model.get_num_layers())
        # remove first and last elements
        layers = np.delete(layers, [0, len(layers) - 1])
    else:
        layers = np.asarray(layers.replace('[', '').replace(']', '').split(',')).astype(int)

    includedLayers = layers  # store original included layers
    isPad = False

    #     flat = np.empty(shape=(len(inData),len(layers))
    #     util.printDebug('flatshape = %s'%(flat.shape))

    util.thisLogger.logInfo('Applying activation extraction to layers: %s' % (layers))

    isFirstLayer = True
    for layerNum in layers:
        # try:
        #     getLastLayerOutput = K.function([model.layers[0].get_input_at(0)],
        #                               [model.layers[layerNum].output])
        # except:
        #     getLastLayerOutput = K.function([model.layers[0].get_input_at(1)],
        #                                     [model.layers[layerNum].output])

        if (layerExtraction == "single"):
            # all values from one layer
            # temp, only using the last activation layer at the moment
            # if layer is -9 it has no global max pooling applied
            # if layer is -8 it has global max pooling applied
            # util.printDebug(layerNum)
            if layerNum in layers:
                # layer_output = getLastLayerOutput([inData])[0]

                if isPad:
                    # check if this layer requires padding
                    if layerNum not in includedLayers:
                        layer_output.fill(0)

                # key = "activation" +str(layerNum)
                key = model.get_layer_name(layerNum)

                # o = getLastLayerOutput([inData])[0]
                #                 util.printDebug(o[0:5])
                #                 util.printDebug(o.shape)

                activationReductionName = util.getParameter("LayerActivationReduction")

                # reset the global variables in the extract module if it is the first layer
                layerResult = singleLayerActivationReduction(key, model.get_layer_output(layerNum, inData),
                                                             reset=isFirstLayer, model=model, inData=inData,
                                                             y_inData=y_inData)
                isFirstLayer = False

                if layerResult.size == 1 and layerResult[0] == 0:
                    util.thisLogger.logInfo(
                        'did not add layer %s as it was removed due to %s calculation' % (key, activationReductionName))
                else:
                    # actDict[key] = layerResult
                    # util.thisLogger.logInfo(layerResult.shape)
                    # util.thisLogger.logInfo('added layer')
                    if layerResults.size == 1 and layerResults[0] == 0:
                        layerResults = layerResult
                    else:
                        layerResults = np.append(layerResults, layerResult, 1)

        else:
            raise ValueError("layerExtraction name of '%s' not recognised: " % (layerExtraction))

    # return actDict
    if inData.shape[0] == 1:
        layerResults = np.reshape(layerResults, (1, layerResults.shape[0]))
    return layerResults
