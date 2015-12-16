# -*- coding: utf-8 -*-
import pandas as pd
import csv
import numpy as np
import cPickle as pickle
from os import path
from dataInfo import columnInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer as DV

def processDiscDataframe(df):    
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

def writeToFile(data, writer):
    for el in data:
        writer.writerow(el)

def saveInputOutput(inA, outA, inFilePath, outFilePath):
    inputFile = csv.writer(open(inFilePath, 'wb'))
    outputFile = csv.writer(open(outFilePath, 'wb'))
    writeToFile(inA, inputFile)
    writeToFile(outA, outputFile)

def file2Data(filePath):
    fileData = []
    with open(filePath, 'rb') as fp:
        reader = csv.reader(fp)
        for row in reader:
            fileData.append(row)
    return fileData

def generateModel(inFilePath, outFilePath):
    inData = file2Data(inFilePath)
    outData = file2Data(outFilePath)
    print len(outData)
    print outData
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(inData, outData)

