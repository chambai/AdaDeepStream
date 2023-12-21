import numpy as np
import util

# helper module for extracting activations
lastHiddenLayer = np.asarray([0])
module = None


# ----------------------------------------------------------------------------
def loadModule(modName, dsData=None):
    global moduleName, module, results
    results = None
    moduleName = modName
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

    # util.thisLogger.logInfo('Applying layer activation reduction: %s for layer %s' % (layerActivationReduction, key))
    for layerReductionName in layerActRedArray:
        loadModule('modules_reduce.' + layerReductionName)
        if layerReductionName == 'pad' or layerReductionName == 'flatten':
            layer = module.extractSingleLayer(key=key, layer=layer, reset=reset)
        else:
            layer = module.extractSingleLayer(key=key, layer=layer, reset=reset,
                                          layerActivationReduction=layerReductionName, model=model, inData=inData,
                                          y_inData=None)
    return np.asarray(layer)


# ----------------------------------------------------------------------------
def layerActivationReduction(actDict):
    layerActivationReduction = util.getParameter('LayerActivationReduction')
    for layerReductionName in layerActivationReduction:
        # util.thisLogger.logInfo('Applying layer activation reduction: %s' % (layerReductionName))
        loadModule('modules_extract.' + layerReductionName)
        actDict = module.extractMultiLayer(actDict)
    return actDict

# -----------------------------------------------------------------------
def getActivations(model, inData, y_inData=None):
    global lastHiddenLayer
    layerResults = np.zeros(1)

    # ActivationLayerExtraction
    layerExtraction = 'single'
    layers = 'all'

    if (layers == 'all'):
        layers = np.arange(model.get_num_layers())
        # remove first and last elements
        layers = np.delete(layers, [0, len(layers) - 1])
    else:
        layers = np.asarray(layers.replace('[', '').replace(']', '').split(',')).astype(int)

    isFirstLayer = True
    for layerNum in layers:
        if (layerExtraction == "single"):
            # all values from one layer
            if layerNum in layers:
                key = model.get_layer_name(layerNum)

                # reset the global variables in the extract module if it is the first layer
                layerResult = singleLayerActivationReduction(key, model.get_layer_output(layerNum, inData),
                                                             reset=isFirstLayer, model=model, inData=inData,
                                                             y_inData=y_inData)
                isFirstLayer = False

                if layerResult.size == 1 and layerResult[0] == 0:
                    pass
                else:
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
