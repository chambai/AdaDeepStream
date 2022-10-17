# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:05:39 2019

@author: id127392
"""
import util
from skmultiflow.data import DataStream
from core import extract, datamanip
import numpy as np
import datetime

moduleName = None
module = None
results = None
instanceProcessingStartTime = None
unadapted_window_processing_time = np.array([])
adapted_window_processing_time = np.array([])

#----------------------------------------------------------------------------
def loadModule(modName, dsData=None):
    global moduleName, module, results
    results = None
    moduleName = modName
    #module = __import__("das.analyse_" + moduleName, fromlist=[''])
    module = __import__(moduleName, fromlist=[''])
    if hasattr(module, 'setData'):
        module.setData(dsData)

#----------------------------------------------------------------------------
def flatSetupDnn(dnn, flatActivations, y_train, in_x_data_train):
    global module, moduleName
    util.thisLogger.logInfo("---------- start of %s setup for activaton analysis with dnn----------"%(moduleName))
    module.setupAnalysis(dnn, flatActivations, y_train, in_x_data_train)
    util.thisLogger.logInfo("----------- end of %s setup for activaton analysis with dnn----------\n"%(moduleName))

#----------------------------------------------------------------------------
def flatSetupCompare(dnn, x_train, y_train):
    global module, moduleName
    util.thisLogger.logInfo("---------- start of %s setup for comparison----------"%(moduleName))
    module.setupAnalysis(dnn, x_train, y_train)
    util.thisLogger.logInfo("----------- end of %s setup for comparison----------\n"%(moduleName))

#----------------------------------------------------------------------------
def processStreamInstances():
    # only used if instances need to be processed on a separate thread
    global module
    module.processStreamInstances()
    temp = 'not implemented'

#----------------------------------------------------------------------------
def stopProcessing():
    # only used if instances need to be processed on a separate thread and you need to stop the thread
    global module, unadapted_window_processing_time, adapted_window_processing_time
    if hasattr(module, 'stopProcessing'):
        module.stopProcessing()
    # calculate timings
    window_size = util.getParameter('StreamBatchLength')
    # calculate average per instance
    unadapted_instance_processing_time = unadapted_window_processing_time / window_size
    avg_unadapted_instance_time = round(np.average(unadapted_instance_processing_time), 3)
    sd_unadapted_instance_time = round(np.std(unadapted_instance_processing_time), 3)
    util.thisLogger.logInfo('AverageUnadaptedInstanceTime (ms)=%s (%s)' % (avg_unadapted_instance_time, sd_unadapted_instance_time))

    avg_adapted_window_time = round(np.average(adapted_window_processing_time), 3)
    sd_adapted_window_time = round(np.std(adapted_window_processing_time), 3)
    util.thisLogger.logInfo('AverageAdaptionWindowTime (s)=%s (%s)' % (avg_adapted_window_time, sd_adapted_window_time))

    unadapted_window_processing_time = np.array([])
    adapted_window_processing_time = np.array([])


#----------------------------------------------------------------------------
def getAnalysisParams():
    # return a list of parameters - optional
    global module
    params = module.getAnalysisParams()
    return params

# ----------------------------------------------------------------------------
def startDataInputStream(dnnModel, simPrediction, maxClassValues1, maxClassValues2, unseenData):

    util.thisLogger.logInfo("\n---------- start of data input stream ----------")
    returnUnseenInstances = startDataInputStream_DataStream(dnnModel, simPrediction, maxClassValues1,maxClassValues2, unseenData)
    util.thisLogger.logInfo("----------- end of data input stream ----------\n")

    return returnUnseenInstances

# ----------------------------------------------------------------------------
def startDataInputStream_DataStream(dnnModel, simPrediction, maxClassValues1, maxClassValues2, unseenData):
    global module, moduleName, instanceProcessingStartTime, unadapted_window_processing_time, adapted_window_processing_time
    batchLength = util.getParameter('StreamBatchLength')
    analysisName = util.getParameter('AnalysisModule')

    # split unseen instance object list into batches
    unseenDataBatches = [unseenData[i:i + batchLength] for i in range(0, len(unseenData), batchLength)]

    dataInstances = np.array([x.instance.flatten() for x in unseenData],dtype=float)

    # pass in zeros for Y data as the dnn predicts will be done later
    stream = DataStream(dataInstances, np.zeros(len(dataInstances)))

    instanceProcessingStartTime = datetime.datetime.now()
    returnUnseenInstances = []
    batchIndex = 1
    instIndex = 0
    while stream.has_more_samples():
        window_start_time = datetime.datetime.now()
        util.thisLogger.logInfo('batch %s' % (batchIndex))

        isLastBatch = len(unseenDataBatches) == batchIndex

        batchDataInstances, batchDnnPredicts = stream.next_sample(batchLength)  # get the next sample
        trueDiscrep = np.array([u.discrepancyName for u in unseenDataBatches[batchIndex - 1]])

        # get the original data shape and shape the batch data instances
        dataShape = unseenData[0].instance.shape  # (1, 32, 32, 3)
        batchDataInstances = np.reshape(batchDataInstances,
                                        (len(batchDataInstances), dataShape[1], dataShape[2], dataShape[3]))

        #if batchDataInstances == None:
        if unseenData[instIndex].reducedInstance.shape[0] == 0 or (unseenData[instIndex].reducedInstance.shape[0] == 1 and unseenData[instIndex].reducedInstance[0] == 0):
            batchDnnPredicts = getPredictions(dnnModel, batchDataInstances)

            if 'compare' in analysisName:
                batchActivations = batchDataInstances
            else:
                batchActivations = online_processInstances(batchDataInstances, dnnModel, simPrediction, maxClassValues1,
                                                           maxClassValues2)
        else:
            # data has already been reduced/normalized etc...
            batchActivations = np.array([u.reducedInstance for u in unseenDataBatches[batchIndex-1]],dtype=float)

        batchTrueLabels = [u.correctResult for u in unseenDataBatches[batchIndex - 1]]

        if 'modules_adapt' in moduleName:
            if 'ocl' in moduleName:
                batchDataInstances = [u.instance for u in unseenDataBatches[batchIndex-1]]
            else:
                batchDataInstances = [u.instance for u in unseenDataBatches[batchIndex - 1]]

            dataShape = batchDataInstances[0].shape
            batchDataInstances = np.reshape(batchDataInstances,
                                            (len(batchDataInstances), dataShape[1], dataShape[2], dataShape[3]))

            batch_result = module.processUnseenStreamBatch(batchDataInstances, batchActivations, batchDnnPredicts, batchTrueLabels, trueDiscrep)
            window_end_time = datetime.datetime.now()
            if (hasattr(module, 'analysis') and module.analysis.is_adapted) or  (hasattr(module, 'is_adapted') and module.is_adapted):
                adapted_window_processing_time = np.append(adapted_window_processing_time, round((window_end_time - window_start_time).total_seconds()))
            else:
                unadapted_window_processing_time = np.append(unadapted_window_processing_time, round((window_end_time - window_start_time).total_seconds()*1000))
        else:
            if 'modules_compare' in util.getParameter('AnalysisModule'):
                batch_result = module.processUnseenStreamBatch(batchDataInstances, batchDnnPredicts, batchTrueLabels)
            else:
                batch_result = module.processUnseenStreamBatch(batchActivations, batchDnnPredicts, batchDataInstances, batchTrueLabels, isLastBatch, trueDiscrep)
            window_end_time = datetime.datetime.now()
            unadapted_window_processing_time = np.append(unadapted_window_processing_time, round((window_end_time - window_start_time).total_seconds()*1000))

        if len(batch_result) == 2:
            batchDiscrepResult = batch_result[0]
            analysis_predicts = batch_result[1]
            adapt_discrepancy = np.empty(len(batchDiscrepResult), dtype=str)
            adapt_discrepancy[:] = "-"
            adapt_class = np.empty(len(batchDiscrepResult), dtype=str)
            adapt_class[:] = "-"
        elif len(batch_result) == 4:
            batchDiscrepResult = batch_result[0]
            analysis_predicts = batch_result[1]
            adapt_discrepancy = batch_result[2]
            adapt_class = batch_result[3]
        else:
            batchDiscrepResult = batch_result
            analysis_predicts = np.empty(len(batchDiscrepResult), dtype=str)
            analysis_predicts[:] = "-"
            adapt_discrepancy = np.empty(len(batchDiscrepResult), dtype=str)
            adapt_discrepancy[:] = "-"
            adapt_class = np.empty(len(batchDiscrepResult), dtype=str)
            adapt_class[:] = "-"


        for i, (act, dnnPredict, res, analysisPredict, adaptDiscrepancy, adaptClass) in enumerate(zip(batchActivations, batchDnnPredicts, batchDiscrepResult, analysis_predicts, adapt_discrepancy, adapt_class)):
            unseenData[instIndex].reducedInstance = np.array(act) #act.reshape(1,act.shape[0])
            unseenData[instIndex].predictedResult = dnnPredict
            unseenData[instIndex].discrepancyResult = res
            unseenData[instIndex].analysisPredict = analysisPredict
            unseenData[instIndex].adaptDiscrepancy = adaptDiscrepancy
            unseenData[instIndex].adaptClass = adaptClass
            if (hasattr(module, 'analysis') and module.analysis.is_adapted) or  (hasattr(module, 'is_adapted') and module.is_adapted):
                unseenData[instIndex].driftDetected = True
            returnUnseenInstances.append(unseenData[instIndex])
            instIndex += 1

        batchIndex += 1

    if hasattr(module, 'logTestAccuracy'):
        ndDataInstances = np.array([x.instance for x in unseenData if x.discrepancyName == 'ND'], dtype=float)
        if len(ndDataInstances) > 0:
            ndActivations = np.array([x.reducedInstance for x in unseenData if x.discrepancyName == 'ND'], dtype=float)
            if dnnModel != None:
                if hasattr(module, 'get_x_data'):
                    ndActivations = module.get_x_data(dnnModel, ndDataInstances, ndActivations)
            ndDnnPredicts = np.array([x.predictedResult for x in unseenData if x.discrepancyName == 'ND'], dtype=int)
            module.logTestAccuracy(ndActivations, ndDnnPredicts)

    if hasattr(module, 'endOfUnseenStream'):
        module.endOfUnseenStream()

    # print out string of results for debug purposes
    classStr = ''
    discrepancyStr = ''
    for inst in returnUnseenInstances:
        classStr += ''.join(str(inst.correctResult))
        discrepancyStr += ''.join(str(inst.discrepancyResult))

    util.thisLogger.logInfo('TRUE_CLASS:    %s' % (classStr))
    util.thisLogger.logInfo('DISCREPANCIES: %s' % (discrepancyStr))

    return np.asarray(returnUnseenInstances)



# ----------------------------------------------------------------------------
def online_processInstances(instances, dnnModel, simPrediction, maxClassValues1, maxClassValues2):
    # processes the instances by getting the predictions and activations from the DNN and reducing them.
    global instanceProcessingStartTime
    instanceProcessingStartTime = datetime.datetime.now()
    util.thisLogger.logInfo('Start of instance processing, %s' % (len(instances)))

    flatActivations = extract.getActivationData2(dnnModel, instances)
    del instances  # delete instances to save memory

    # Normalize raw flat activations
    if maxClassValues1 != None:
        util.thisLogger.logInfo('Max value of raw activations: %s' % (maxClassValues1))
        flatActivations, maxClassValues1 = datamanip.normalizeFlatValues(flatActivations, False, maxClassValues1)

    # normalise the reduced instance activations
    if maxClassValues2 != None:
        util.thisLogger.logInfo('Max value of reduced activations: %s' % (maxClassValues2))
        flatActivations, maxClassValues2 = datamanip.normalizeFlatValues(flatActivations, False, maxClassValues2)

    return flatActivations

# ----------------------------------------------------------------------------
def getPredictions(dnnModel, instances, classes=[]):
    predictions = dnnModel.predict(instances)

    # The DNN is trained to output 0 or 1 only.
    mapNewYValues = util.getParameter('MapNewYValues')
    if len(mapNewYValues) == 0:
        # get the original classes it was trained on and transform the outputs
        if len(classes) == 0:
            classes = util.getParameter('DataClasses')
        util.thisLogger.logInfo('Data classes to be used: %s' % (classes))
        predictions = util.transformZeroIndexDataIntoClasses(predictions, classes)
    else:
        # get the mapped values the DNN was trained on
        mapNewYValues = np.unique(mapNewYValues)
        mapNewYValues = mapNewYValues[mapNewYValues >= 0]
        util.thisLogger.logInfo('Mapped class values to be used: %s' % (mapNewYValues))

    return predictions






