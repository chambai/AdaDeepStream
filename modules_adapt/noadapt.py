import numpy as np
import util
import threading


adaptive_dnn = None
stop_thread = False
adapt_lock = threading.Lock()
model_lock = threading.Lock()
data_lock = threading.Lock()
is_adapting = False
has_updated_dnn = False
add_adaption_layers = True
final_layer_act_only = False
limit_adaption_layers = False
x_act_buffer = np.array([0])
x_data_buffer = np.array([0])
y_data_buffer = np.array([0])

def setData(inDsData):
    pass

# no drift detection, no dnn adaption - DNN is already trained on all classes
def setupAnalysis(indnn, act_train_batch, y_train_batch, inData=[]):
    pass

def processUnseenStreamBatch(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, true_values, true_discrep):
    # batch from the stream is provided to this function
    true_label_str = ''
    dnn_predict_str = ''
    for d, t in zip(dnnPredict_batch, true_values):
        true_label_str += str(int(t))
        dnn_predict_str += str(int(d))
    util.thisLogger.logInfo('True_label:   %s' % (true_label_str))
    util.thisLogger.logInfo('DNN_Predict:  %s' % (dnn_predict_str))
    util.thisLogger.logInfo('')

    result = ['N' for i in range(len(act_unseen_batch))]
    adapt_result = ['ND' for i in range(len(act_unseen_batch))]

    return result, dnn_predict_str, adapt_result, dnn_predict_str

def processStreamInstances():
    pass

def stopProcessing():
    pass

def getAnalysisParams():
    parameters = []
    parameters.append(['none',0])
    return parameters
