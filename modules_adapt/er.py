import util
from external.ocl.general_main_dsd import OclMgr
from modules_adapt.base_ocl import AnalysisOcl

analysis = None

def setData(inDsData):
    pass

def setupAnalysis(indnn, act_train_batch, y_train_batch, inData=[]):
    global analysis
    analysis = AnalysisOcl()
    analysis.sequential_adapt_override = True
    accuracies = analysis.setup_activation_classifiers(act_train_batch, y_train_batch)
    analysis.setup_drift_detectors()
    x_train_orig, y_train_orig = analysis.get_training_data()
    indnn.model = analysis.set_dnn_output_features(indnn.model, 10) # have to give the total number of classes is in the dataset
    lr, wd = analysis.get_optimizer_params()

    ocl_mgr = OclMgr(indnn.model, 'ER', util.getParameter('DatasetName'), util.getParameter('Dnn'), util.getParameter('DataClasses'), accuracies.max(), lr, wd)
    ocl_mgr.fit(x_train_orig, y_train_orig)
    analysis.adaptive_dnn = ocl_mgr


def logTrainAccuracy(act_train, y_train):
    global analysis
    analysis.logTrainAccuracy(act_train, y_train)

def logTestAccuracy(act_unseen, y_unseen):
    global analysis
    analysis.logTestAccuracy(act_unseen, y_unseen)

def processUnseenStreamBatch(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, true_values, true_discrep):
    global analysis
    return analysis.processUnseenStreamBatch(xdata_unseen_batch, act_unseen_batch, dnnPredict_batch, true_values, true_discrep)

def processStreamInstances():
    pass

def stopProcessing():
    global analysis
    analysis.stopProcessing()

def getAnalysisParams():
    return analysis.getAnalysisParams()
