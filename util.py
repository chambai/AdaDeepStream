import logging
import csv
import datetime
import os
import os.path, time
import threading
import numpy as np
from sklearn import preprocessing

paramDefFile = 'input/params/paramdef.txt'
setupFile = None
thisLogger = None
isLoggingEnabled = True
logPath = 'output/log.log'
logLevel = 'INFO'
params = None
usedParams = []

# ------------------------------------------------------------------------
def getAllParameters():
    global params
    if (params == None):
        # load param def files
        params = getParamData(paramDefFile, setupFile)


# ------------------------------------------------------------------------
def getParamData(paramDefFile, setupFile):
    params = {}
    paramValue = None
    paramdef = getFileLines(paramDefFile)

    # get param names from def file
    paramNames = []
    for param in paramdef:
        splitParams = param.split("#")
        paramName = splitParams[0].strip()
        paramNames.append(paramName)

    # get param values from setup file
    setup = getFileLines(setupFile)
    paramValues = {}
    setup = [s for s in setup if "#" not in s]
    for paramName in paramNames:
        for param in setup:
            splitParams = param.split("=")
            setupName = splitParams[0].strip()
            if paramName == setupName:
                paramValues[paramName] = splitParams[1].strip()

    # store params in dictionary
    for param in paramdef:
        splitParams = param.split("#")
        if len(splitParams) == 2:
            paramType = 'string'
            paramComment = splitParams[1].strip()
        else:
            paramType = splitParams[1].strip()
            paramComment = splitParams[2].strip()

        # check that all parameters that are defined in param def file are in the setup file
        checkAllParamsInSetupFile = True
        if checkAllParamsInSetupFile:
            paramName = splitParams[0].strip()
            if paramName in paramValues:
                paramValue = paramValues[paramName]
            else:
                raise ValueError(
                    'Variable: ' + paramName + ' is defined in param def but does not exist in the setup file: ' + setupFile)
        else:
            paramValue = paramValues[paramName]

        params[paramName] = (paramType, paramComment, paramValue)

    return params

# ------------------------------------------------------------------------
def setParameter(paramName, paramValue):
    global setupFile
    lines = getFileLines(setupFile)
    for n, l in enumerate(lines):
        splitLine = l.split('=')
        if splitLine[0] == paramName:
            lines[n] = '%s=%s\n' % (splitLine[0], paramValue)
            break;
    saveFileLines(setupFile, lines, 'w')

#------------------------------------------------------------------------
def getFileLines(filename):
    file = open(filename, "r")
    lines = file.readlines()
    return lines

#------------------------------------------------------------------------
def saveReducedData(filename, flatActivations, y_train):
    classes = np.unique(y_train)

    # create a csv file for each class
    csvFileNames = []
    for dataClass in classes:
        csvFileName = "%s_%s.csv" % (filename, dataClass)
        filteredActivations = []
        for index in range(len(flatActivations)):
            if (y_train[index] == dataClass):
                filteredActivations.append(flatActivations[index])

        if len(filteredActivations) > 0:
            labels = np.arange(len(filteredActivations[0]))
            filteredActivations = np.concatenate(([labels], filteredActivations), axis=0)  # axis=0 means add rows
            thisLogger.logInfo("saving reduced activations to csv file %s" % (csvFileName))
            saveToCsv(csvFileName, filteredActivations)
            csvFileNames.append(csvFileName)

# ------------------------------------------------------------------------
def getParameter(paramName):
    global usedParams
    global params
    # paramValue = None

    # get parameter values
    getAllParameters()

    # get parameter type and value
    paramType = params[paramName][0]
    if paramType == 'bool':
        paramValue = stringToBool(params[paramName][2])
    elif paramType == 'float':
        paramValue = float(params[paramName][2])
    elif paramType == 'int':
        paramValue = int(params[paramName][2])
    elif paramType == 'string':
        paramValue = params[paramName][2]
    elif paramType == 'stringarray':
        paramValue = []
        splitParams = params[paramName][2].split(',')
        for p in splitParams:
            paramValue.append(p.strip())
    elif paramType == 'intarray':
        if params[paramName][2] == '[]':
            paramValue = []
        else:
            paramValue = np.asarray(params[paramName][2].replace('[', '').replace(']', '').replace('-',',').split(',')).astype(int)
    elif paramType == 'floatarray':
        if params[paramName][2] == '[]':
            paramValue = []
        else:
            paramValue = np.asarray(params[paramName][2].replace('[', '').replace(']', '').split(',')).astype(float)
    else:
        # The value must be one of the values specified
        allowedValues = paramType.split(',')
        allowedValues = [x.strip() for x in allowedValues]  # List comprehension
        paramValue = params[paramName][2]
        if paramValue not in allowedValues:
            print(paramValue + ' is not in the list of allowed values for parameter ' + paramName)
            print('Allowed values are: ' + paramType)

    p = '%s=%s' % (paramName, str(paramValue))
    if not any(p in u for u in usedParams):
        usedParams.append('%s=%s' % (paramName, str(paramValue)))

    return paramValue

# ------------------------------------------------------------------------
def saveFileLines(filename, lines, mode='x', wait=False):
    if wait == False:
        with open(filename, mode) as file:
            file.writelines(lines)
    return lines

# ------------------------------------------------------------------------
def stringToBool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError

# ------------------------------------------------------------------------
def getFileCreationTime(filename):
    return time.ctime(os.path.getctime(filename))


# ------------------------------------------------------------------------
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]

# ------------------------------------------------------------------------
def normalize(activations):
    max_values = []
    if len(activations.shape) == 1:
        maxValue = np.amax(activations)
        activations = activations / maxValue
        max_values.append(maxValue)
    else:
        # list of activations
        for n, a in enumerate(activations[0]):
            maxValue = np.amax(a)
            activations[0][n] = a / maxValue
            max_values.append(maxValue)
    return activations, max_values

# ------------------------------------------------------------------------
def normalizeFlatValues(flatActivations, isTrainingData, maxValue=None):
    # divides each element by the max value of the training data

    # flatActivations = np.vstack(flatActivations)
    # for i in range(len(flatActivations)):
    #    activationMatrix = np.vstack((activationMatrix,flatActivations[i]))

    # util.printDebug(activationMatrix)
    if isTrainingData:
        # if len(flatActivations.shape) > 1:
        if getParameter('DataDiscrepancy') == 'CD':
            block_max_values = []
            for col in range(flatActivations.shape[1]):
                block = flatActivations[:, col]
                if len(block.shape) == 1:
                    bf = block.reshape((block.shape[0]))
                    bf = np.array([b for b in bf], dtype=float)
                else:
                    bf = block
                block_max = np.max(bf)
                block_max_values.append(block_max)
            maxValue = block_max_values
        else:
            maxValue = [np.amax(flatActivations)]

        thisLogger.logDebug('Max Value: %s' % (maxValue))

    dataNormalization = 'norm'
    # dataNormalization = getParameter("DataNormalization")
    if len(maxValue) == 1:
        if dataNormalization == 'norm':
            flatActivations = flatActivations / maxValue
            # thisLogger.logInfo("Normalization (%s) applied" % (dataNormalization))
        elif dataNormalization == 'std':
            flatActivations = preprocessing.scale(flatActivations)
            # thisLogger.logInfo("Standardization (%s) applied" % (dataNormalization))
        elif dataNormalization == 'none':
            thisLogger.logInfo("No data normalisation/standardization")
        else:
            thisLogger.logInfo("unhandled data normalization of %s" % (dataNormalization))
    else:
        for col in range(flatActivations.shape[1]):
            flatActivations[:, col] = flatActivations[:, col] / maxValue[col]

    # for i in range(len(flatActivations)):
    #    flatActivations[i] = activationMatrix[:i]

    # util.printDebug(activationMatrix)
    # util.printDebug(flatActivations)

    flatActivations = np.nan_to_num(flatActivations, copy=False)

    return flatActivations, maxValue

def getExperimentName():
    dnn = getParameter('Dnn')
    datasetName = getParameter('DatasetName')
    classes = np.unique(getParameter('DataClasses'))
    classesName = ''.join(map(str, classes))
    num_classes = len(classes)

    # make hashcode if there's too many classes
    if num_classes > 10:
        hash_classes = hash(tuple(classes))  # hash classes to get unique number otherwise name is too long
        classesName = str(hash_classes)

    # add postfix if sub if there's sub-classes incase there's a clash with previous super class names
    if has_sub_classes():
        classesName = classesName + '_sub'

    exp_name = '%s_%s_%s' % (dnn, datasetName, classesName)
    return exp_name

def transformZeroIndexDataIntoClasses(y_data, classes):
    # changes y data that is from 0 to n into the classes
    # Elements in y_data that are 0 will be changed to the first element in the classes list,
    # elements in y_data that are 1 will be changed to the second element in the classes list etc...
    y_data_new = np.copy(y_data)
    count = 0
    for c in classes:
        for i, y in enumerate(y_data):
            if y == count:
                y_data_new[i] = c
        count += 1
    y_data = y_data_new
    del y_data_new
    return y_data

# ------------------------------------------------------------------------
def transformDataIntoZeroIndexClasses(y_data, originalClasses=[], newClasses=[]):
    # changes y data that is from 0 to n into the classes
    # Elements in y_data that are 0 will be changed to the first element in the classes list,
    # elements in y_data that are 1 will be changed to the second element in the classes list etc...
    # originalClasses = np.sort(np.unique(y_data))
    y_data_new = np.copy(y_data)
    count = 0
    for c in originalClasses:
        for i, y in enumerate(y_data):
            if y == c:
                if len(newClasses) > 0:
                    y_data_new[i] = newClasses[np.array(originalClasses).tolist().index(c)]
                else:
                    y_data_new[i] = count  # no classes specified, assign classes sequentially from zero
        count += 1
    y_data = y_data_new
    del y_data_new
    return y_data


# ------------------------------------------------------------------------
def filterDataByClass(x_data, y_data, class_array):
    ix = np.isin(y_data, class_array)
    ixArry = np.where(ix)
    indexes = ixArry[0]  # list of indexes that have specified classes
    x_data = x_data[indexes]
    y_data = y_data[indexes]
    # print('classarray: %s, x_data: %s'%(class_array, len(x_data)))
    # print('classarray: %s, y_data: %s'%(class_array, len(y_data)))
    return x_data, y_data

csvFileLock = threading.Lock()
# ------------------------------------------------------------------------
# saves a vector to a csv file
def saveToCsv(csvFilePath, vector, append=False):
    csvFileLock.acquire()
    fileWriteMethod = 'w'
    if append == True:
        fileWriteMethod = 'a'

    with open(csvFilePath, fileWriteMethod, newline='') as csv_file:
        csvWriter = csv.writer(csv_file, delimiter=',')
        csvWriter.writerows(vector)

    now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print("%s : saved" % (now))
    csvFileLock.release()

# ------------------------------------------------------------------------
def getColourText(text, colour):
    printCode = None
    if (colour == 'red'):
        printCode = "\x1b[31m" + text + "\x1b[0m"
    elif (colour == 'green'):
        printCode = "\x1b[32m" + text + "\x1b[0m"
    elif (colour == 'blue'):
        printCode = "\x1b[34m" + text + "\x1b[0m"
    elif (colour == 'magenta'):
        printCode = "\x1b[35m" + text + "\x1b[0m"
    elif (colour == 'cyan'):
        printCode = "\x1b[36m" + text + "\x1b[0m"
    elif (colour == 'black'):
        printCode = "\x1b[30m" + text + "\x1b[0m"
    else:
        raise ValueError('Colour: ' + colour + ' is not a recognised colour')

    return printCode

# ------------------------------------------------------------------------
class Logger:
    def __init__(self):  # double underscores means the function is private
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        global logLevel
        if (logLevel == 'INFO'):
            logging.basicConfig(filename=logPath, level=logging.INFO)
        else:
            logging.basicConfig(filename=logPath, level=logging.DEBUG)

    # class methods always take a class instance as the first parameter
    def logInfo(self, item):
        show_logs = True
        if show_logs:
            global logLevel
            item = prefixDateTime(item)
            if (logLevel == 'INFO'):
                print(item)
                # print("\x1b[31m\"red\"\x1b[0m")
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.info(item)

    def logInfoColour(self, item, colour):
        global logLevel
        item = prefixDateTime(item)
        if (logLevel == 'INFO'):
            print(getColourText(item, colour))
            # print("\x1b[31m\"red\"\x1b[0m")
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.info(item)

    def logDebug(self, item):
        global logLevel
        item = prefixDateTime(item)
        if (logLevel == 'DEBUG'):
            print(item)
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.debug(item)

    def logError(self, item):
        item = prefixDateTime(item)
        print(item)
        global isLoggingEnabled
        if (isLoggingEnabled == True):
            logging.error(item)

    def closeLog(self):
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)


# ------------------------------------------------------------------------
def prefixDateTime(item):
    if item:
        item = '%s %s' % (datetime.datetime.now(), item)
    return item

def get_mapping_param(param_name):
    param_value = ''
    dataset_name = getParameter('DatasetName')
    discrep = getParameter('DataDiscrepancy')
    if discrep == 'CD':
        if dataset_name == 'cifar10':
            # cifar
            if param_name == 'MapOriginalYValues':
                param_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            elif param_name == 'MapNewYValues':
                param_value = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
            elif param_name == 'MapNewNames':
                param_value = ['transport', 'animal']
            else:
                raise Exception('unhandled param name of %s'%(param_name))

        elif dataset_name == 'cifar100':
            # cifar
            if param_name == 'MapOriginalYValues':
                param_value = np.arange(100).tolist()
            elif param_name == 'MapNewYValues':
                param_value = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
            elif param_name == 'MapNewNames':
                param_value = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                              'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                              'large man-made outdoor things', 'large natural outdoor scenes',
                              'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates',
                              'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']
            else:
                raise Exception('unhandled param name of %s'%(param_name))

        elif dataset_name == 'mnistfashion':
            # fashion
            if param_name == 'MapOriginalYValues':
                param_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            elif param_name == 'MapNewYValues':
                param_value = [1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
            elif param_name == 'MapNewNames':
                param_value = ['footwear', 'clothes']
            else:
                raise Exception('unhandled param name of %s'%(param_name))
        else:
            raise Exception('unhandled dataset of %s'%(dataset_name))
    elif discrep == 'CE':
        if param_name == 'MapNewNames':
            param_value = 'none'
        else:
            param_value = []
    else:
        raise Exception('unhandled discrep of %s' % (discrep))

    return param_value

def has_sub_classes():
    return len(get_mapping_param('MapNewYValues')) > 0

# -------------------------------------------------------------------------
def map_classes_y(y):
    y_unmapped = y

    # maps data to higher level classes
    mapOriginalYValues = get_mapping_param('MapOriginalYValues')

    # map data if mapOriginalYValues contains data
    if len(mapOriginalYValues) != 0:
        mapNewYValues = np.asarray(get_mapping_param('MapNewYValues'))
        mapNewNames = get_mapping_param('MapNewNames')

        # check mapOriginalYValues and mapNewYValues are the same size
        if len(mapOriginalYValues) != len(mapNewYValues):
            raise ValueError("MapOriginalYValues array size (%s) does not match MapNewYValues array size (%s)" % (
            len(mapOriginalYValues), len(mapNewYValues)))

        # check distinct values of mapNewYValues match number of elements in mapNewNames
        distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
        if len(distinctMapNewYValues) != len(mapNewNames):
            raise ValueError(
                "Distinct values of MapNewYValues (%s) does not match the number of elements in MapNewNames (%s)" % (
                len(distinctMapNewYValues), len(mapNewNames)))

        # if there's any -1 values in mapNewYValues, remove X and Y values for the corresponding class in mapOriginalYValues
        if -1 in mapNewYValues:
            # find out what elements in mapOriginalYValues the -1 corresponds to
            minusOneIndexes = np.where(mapNewYValues == -1)
            yValuesToRemove = mapOriginalYValues[minusOneIndexes]
            dataIndexesToRemove = np.in1d(y, yValuesToRemove).nonzero()[0]
            y = np.delete(y, dataIndexesToRemove, axis=0)
            y_unmapped = y

        y_dict = {}
        for orig, new in zip(mapOriginalYValues, mapNewYValues):
            y_dict[orig] = new

        y_out = []
        y = np.reshape(y, (-1))
        for y_val in y:
            y_out.append(y_dict[y_val])
        y_out = np.array(y_out)

    return y_out, y_unmapped

# -------------------------------------------------------------------------
def mapClasses(x, y):
    y_unmapped = y

    # maps data to higher level classes
    mapOriginalYValues = get_mapping_param('MapOriginalYValues')

    # map data if mapOriginalYValues contains data
    if len(mapOriginalYValues) != 0:
        thisLogger.logInfo('Mapping classes: length of x data: %s. Length of y data: %s. Y data values: %s' % (
        len(x), len(y), np.unique(y)))
        mapNewYValues = np.asarray(get_mapping_param('MapNewYValues'))
        mapNewNames = get_mapping_param('MapNewNames')

        # check mapOriginalYValues and mapNewYValues are the same size
        if len(mapOriginalYValues) != len(mapNewYValues):
            raise ValueError("MapOriginalYValues array size (%s) does not match MapNewYValues array size (%s)" % (
            len(mapOriginalYValues), len(mapNewYValues)))

        # check distinct values of mapNewYValues match number of elements in mapNewNames
        distinctMapNewYValues = np.unique(mapNewYValues[mapNewYValues >= 0])
        if len(distinctMapNewYValues) != len(mapNewNames):
            raise ValueError(
                "Distinct values of MapNewYValues (%s) does not match the number of elements in MapNewNames (%s)" % (
                len(distinctMapNewYValues), len(mapNewNames)))

        # if there's any -1 values in mapNewYValues, remove X and Y values for the corresponding class in mapOriginalYValues
        if -1 in mapNewYValues:
            # find out what elements in mapOriginalYValues the -1 corresponds to
            minusOneIndexes = np.where(mapNewYValues == -1)
            yValuesToRemove = mapOriginalYValues[minusOneIndexes]
            dataIndexesToRemove = np.in1d(y, yValuesToRemove).nonzero()[0]
            y = np.delete(y, dataIndexesToRemove, axis=0)
            y_unmapped = y
            x = np.delete(x, dataIndexesToRemove, axis=0)

        y_dict = {}
        for orig, new in zip(mapOriginalYValues, mapNewYValues):
            y_dict[orig] = new

        y_out = []
        y = np.reshape(y, (-1))
        for y_val in y:
            y_out.append(y_dict[y_val])
        y_out = np.array(y_out)

        # thisLogger.logInfo('Mapped classes: length of x data: %s. Length of y data: %s. Y data values: %s' % (
        # len(x), len(y_out), np.unique(y_out)))
    return x, y_out, y_unmapped


