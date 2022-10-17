import util
import numpy as np

def extractSingleLayer(key, layer, reset=False):
    # pads 1D array with zeros to specified length
    # pad the array with zero's to a specified number of elements
    # expects the data to be of shape (instances,flat activations)
    value = layer
    paddedArray = value
    padLength = np.asarray(util.getParameter('PadLength'))

    if padLength > value.shape[1]:
        # pad with zero's to make them all the same size
        shape = np.shape(value)
        paddedArray = np.zeros((value.shape[0], padLength))
        paddedArray[:shape[0], :shape[1]] = value
    elif padLength == value.shape[1]:
        util.thisLogger.logInfo('No padding required for layer %s, as shape of %s equals padding length of %s' % (
        key, value.shape, padLength))
    else:
        util.thisLogger.logInfo(
            'Could not apply padding to layer %s, as data length of %s exceeds padding length of %s' % (
            key, value.shape[1], padLength))

    return paddedArray