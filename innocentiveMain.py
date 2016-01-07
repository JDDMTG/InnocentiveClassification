# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo

def preprocess(fileName, train=True):
    df = dm.file2Dataframe(fileName)
    inputArr, outputArr = dm.processDataframeForNP(df, train)
    inputFileName, outputFileName = dm.makeFileName(fileName)
    
    if train:
        dm.save(outputFileName, outputArr)
        
    dm.save(inputFileName, inputArr)
    
    return inputArr, outputArr

print compInfo.dataPoints500File
inA, outA = preprocess(compInfo.dataPoints500File)

print inA.shape
print outA.shape

loadedInA = dm.load(compInfo.outputDataDirectory + 'Innocentive_500_Sample_input.np')
loadedOutA = dm.load(compInfo.outputDataDirectory + 'Innocentive_500_Sample_output.np')

print loadedInA.shape
print loadedOutA.shape

# fullDataIn, fullDataOut = preprocess(compInfo.inputDataPath)
testDataIn, testDataOut = preprocess(compInfo.originalDataDirectory + compInfo.testingInputPathCSV, False)
