# -*- coding: utf-8 -*-
import dataMngr as dm

def preprocess(fileName, train=True):
    df = dm.file2Dataframe(fileName)
    inputArr, outputArr = dm.processDataframeForNP(df, train)
    inputFileName, outputFileName = dm.makeFileName(fileName)
    
    if train:
        dm.save(outputFileName, outputArr)
        
    dm.save(inputFileName, inputArr)
    
    return inputArr, outputArr
    
testFile = '/Users/JuanDa/Google Drive/Innocentive Marketing ML Challenge/Innocentive_500_Sample.csv'

inA, outA = preprocess(testFile)

print inA.shape
print outA.shape