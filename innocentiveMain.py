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
    
testFile = '/Users/user/Documents/Innocentive_Challenge/Innocentive_500_Sample.csv'

inA, outA = preprocess(testFile)
dm.saveInputOutput(inA, outA, '../testin.csv', '../testout.csv')
dm.generateModel('../testin.csv', '../testout.csv')


# print inA.shape
# print outA.shape