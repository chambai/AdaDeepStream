# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:50:23 2019

@author: id127392
"""
import sys
import logging
import csv
import datetime
import os
import platform
import subprocess
import os.path, time
import threading
import numpy as np

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

# ----------------------------------------------------------------------------
def hasSubClasses():
    mapNewYValues = getParameter('MapNewYValues')
    if len(mapNewYValues) == 0:
        return False
    else:
        return True

# ------------------------------------------------------------------------
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


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



