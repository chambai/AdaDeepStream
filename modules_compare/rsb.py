import util
from external.rsb.benchmark.er_runner_ds import DeepStreamRunner
from core.dnn_models import DNNPytorchAdaptive
import numpy as np

dsr = None
is_adapted = True # set to true as each cll adapts the DNN in this method

def setupAnalysis(dnn, param_x_train_batch, param_y_train_batch):
    global dsr
    dnn_pt = DNNPytorchAdaptive()
    opt = dnn_pt.get_optimizer(dnn.model, util.getParameter('Dnn'))
    dsr = DeepStreamRunner(dnn.model, opt)
    dsr.fit(param_x_train_batch, param_y_train_batch)

def processUnseenStreamBatch(batchDataInstances, batchTrueLabels):
    global dsr
    _, predictions = dsr.partial_fit(batchDataInstances, np.array(batchTrueLabels))

    batchDiscrepResult = []
    analysis_predicts = predictions
    adapt_discrepancy = []
    adapt_class = predictions

    true_labels_str = ''
    predict_str = ''
    for t, p in zip(batchTrueLabels, predictions):
        true_labels_str += str(t)
        predict_str += str(p)
    util.thisLogger.logInfo('TRUE LABELS: %s' % (true_labels_str))
    util.thisLogger.logInfo('PREDICTED:   %s' % (predict_str))
    correct = [y for i, y in enumerate(predictions) if batchTrueLabels[i] == int(y)]
    acc = len(correct) / len(batchTrueLabels)
    util.thisLogger.logInfo('DNN_Predict Acc: %s' % (acc))

    for i in range(len(batchDataInstances)):
        batchDiscrepResult.append('N')
        adapt_discrepancy.append('N')

    return batchDiscrepResult, analysis_predicts, adapt_discrepancy, adapt_class

def getAnalysisParams():
    parameters = []
    # return a list of variable parameters in the format [[name,value],[name,value]...[name,value]]
    parameters.append(['none',0])
    return parameters