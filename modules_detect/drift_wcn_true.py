import util

# checks if there is a change, if there is it sets the discrepancy values to teh true values
# (using the true values replaces the human labelling)
def convertDriftToDiscrepancies(batchDrift, trueDiscrep):
    discrepancyStr = ''.join(['N']*len(batchDrift))
    if 'W' in batchDrift:
        discreps = []
        for d in trueDiscrep:
            if d == 'ND':
                discreps.append('N')
            else:
                discreps.append('D')
        discrepancyStr = ''.join([str(t) for t in discreps])

    if util.thisLogger == None:
        print('DISCR_Output: %s' % (discrepancyStr))
    else:
        util.thisLogger.logInfo('DISCR_Output: %s' % (discrepancyStr))

    return list(discrepancyStr)


