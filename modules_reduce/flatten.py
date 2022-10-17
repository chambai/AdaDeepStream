import util

def extractSingleLayer(key, layer, reset=False):
    # flattens data from entire layer into 1D Array
    # flatten the layer to 2D (instances,activations)
    value = layer
    if value is None:
        util.thisLogger.logInfo('Could not flatten layer as it has value of (%s): %s' % (key, value))
    else:
        dataShapeLength = len(value.shape)
        if dataShapeLength == 2:
            layer = value.reshape(value.shape[0], value.shape[1])
        if dataShapeLength == 3:
            layer = value.reshape(value.shape[0], value.shape[1] * value.shape[2])
        if dataShapeLength == 4:
            layer = value.reshape(value.shape[0], value.shape[1] * value.shape[2] * value.shape[3])

    return layer

