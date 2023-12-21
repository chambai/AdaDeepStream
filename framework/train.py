from framework import extract
import numpy as np
import datetime
import util

# helper module to train DeepStream with activation data and remove incorrect training instances
# ----------------------------------------------------------------------------
def getActivations(x_train, numActivationTrainingInstances, dnnModel, y_train):
    util.thisLogger.logInfo("start of activation data extraction from training data")
    startTime = datetime.datetime.now()
    classes = np.sort(np.unique(util.getParameter('DataClasses')))

    # Only get activations from the instances that are correctly classified
    y_predict = dnnModel.predict(x_train)
    util.thisLogger.logDebug('y_predict: %s' % (y_predict))

    dnnModel.get_layer_names()

    y_train_mapped = y_train

    incorrectPredictIndexes = []
    for i in range(0, len(y_predict) - 1):
        if (y_predict[i] != y_train_mapped[i]):
            incorrectPredictIndexes.append(i)

    if numActivationTrainingInstances == -1:
        numActivationTrainingInstances = len(x_train)

    # Delete incorrect instances from the training data
    y_train_unique = np.sort(np.unique(y_train))
    util.thisLogger.logInfo('TrainingInstanceClasses=%s' % (y_train_unique))
    x_train = np.delete(x_train, incorrectPredictIndexes, axis=0)
    y_train = np.delete(y_train, incorrectPredictIndexes, axis=0)

    # Check that all classes appear in the training data. If not, this can skew the results
    y_train_unique = np.sort(np.unique(y_train))
    util.thisLogger.logInfo('TrainingInstanceClassesIncorrectRemoved=%s' % (y_train_unique))
    if not util.has_sub_classes() and not np.array_equal(classes, y_train_unique):
        raise Exception('Missing classes in the training data. Expected: %s, got %s' % (classes, y_train_unique))

    correctTrainingInstances = len(x_train)
    diff = numActivationTrainingInstances - correctTrainingInstances
    util.thisLogger.logInfo('TotalTrainingInstances=%s' % (numActivationTrainingInstances))
    util.thisLogger.logInfo('CorrectTrainingInstances=%s' % (correctTrainingInstances))
    util.thisLogger.logInfo('%s instances removed from training data due to incorrect prediction' % (diff))
    acc = correctTrainingInstances / numActivationTrainingInstances
    util.thisLogger.logInfo('TrainingDataAcc=%s' % (acc))

    # train in batches
    activationTrainingBatchSize = util.getParameter('ActivationTrainingBatchSize')

    xData = x_train[:numActivationTrainingInstances, ]
    yData = y_train[:numActivationTrainingInstances, ]
    batchData = list(util.chunks(xData, activationTrainingBatchSize))
    y_batchData = list(util.chunks(yData, activationTrainingBatchSize))

    numBatches = len(batchData)
    allactivations = None
    for batchIndex in range(numBatches):
        batch = batchData[batchIndex]
        y_batch = y_batchData[batchIndex]
        util.thisLogger.logInfo("Training batch " + str(batchIndex + 1) + " of " + str(len(batchData)) + " (" + str(
            len(batch)) + " instances)")
        # Get activations and set up streams for the training data
        # get reduced activations for all training data in one go

        # Train in a loop
        util.thisLogger.logInfo(str(len(batch)) + " instances selected from training data")

        activations = extract.getActivations(dnnModel, batch, y_batch)
        if allactivations is None:
            allactivations = activations
        else:
            allactivations = np.append(allactivations, activations, 0)

    endTime = datetime.datetime.now()
    util.thisLogger.logInfo('Total training time: ' + str(endTime - startTime))
    util.thisLogger.logInfo("------- end of activation data extraction for training data --------")
    util.thisLogger.logInfo("")

    return allactivations, xData, yData

