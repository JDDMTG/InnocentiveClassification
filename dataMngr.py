# -*- coding: utf-8 -*-
import pandas as pd
import csv
import numpy as np
import cPickle as pickle
from os import path
from dataInfo import columnInfo

def processDiscDataframe(df):    
    from sklearn.feature_extraction import DictVectorizer as DV
    vectorizer = DV( sparse = False )
    df = df.to_dict('records')
    numpyData = vectorizer.fit_transform(df)
    return numpyData

def processDataframeForNP(df, train=True):
    df = df.drop(columnInfo['extraneous'], axis=1) #Unecessary Info
    
    outputData = None
    if train:
        outputData = df[columnInfo['target']]
        outputData = outputData.as_matrix()
    df.drop(columnInfo['target'], axis=1)
    
    inputDiscrete = df[columnInfo['discrete']]
    inputContinuous = df[columnInfo['continous']]
    
    inputDiscrete = processDiscDataframe(inputDiscrete)
    inputContinuous = inputContinuous.as_matrix()
    
    inputData = np.concatenate(( inputContinuous, inputDiscrete ), axis=1)
    return inputData, outputData

def file2Dataframe(fileName):
    df = pd.read_csv(fileName, engine='c')
    return df

def save(fileName, obj):
    pickle.dump( obj, open( fileName, "wb" ) )
    pass

def load(fileName):
    pickle.load( open( fileName, "rb" ) )
    pass

def seperateTestInputOutput(df):
    pass

def makeFileName(inFileName, train=True):
    base=path.basename(inFileName)
    fileName, ext = path.splitext(base)
    
    outputFileName = None
    if train:
        outputFileName = fileName + '_output.np'
    
    inputFileName = fileName + '_input.np'
    
    return inputFileName, outputFileName

def dataToInputOutput(fileName):
    pathDir = "../" # must be run in the directory where the file is
    with open(pathDir + fileName, 'rb') as fullFile:
        reader = csv.reader(fullFile)
        inputFile = csv.writer(open(pathDir + "Input_" + fileName, 'wb'))
        outputFile = csv.writer(open(pathDir + "Output_" + fileName, 'wb'))
        for row in reader:
            inputRow = []
            for x in xrange(len(row)):
                if x == 1: # where output is located
                    outputFile.writerow(row[x])
                    print row[x]
                else:
                    inputRow.append(row[x])
                if x == (len(row) - 1):
                    inputFile.writerow(inputRow)

#
# dataToInputOutput('Innocentive_500_Sample.csv')
