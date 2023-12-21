from modules_reduce import cbir


def extractSingleLayer(key, layer, reset=False, layerActivationReduction='cbirav8pNone32', model=None, inData=None, y_inData=None):
    # cbir chunking each block output into 16 sections and taking the average and splitting the final hidden layer into 32
    # chunks and averaging each chunk.  Total of 112 values per instance
    cbir.num_channel_chunks = 8
    cbir.num_final_hidden_layer_chunks = 32
    cbir.use_pool_layer_threshold = False
    return cbir.extractSingleLayer(key, layer, reset, layerActivationReduction, model, inData, y_inData)


def extractMultiLayer(actDict, layerActivationReduction='jsdiverge', reset=False):
    # not implemented
    return actDict




