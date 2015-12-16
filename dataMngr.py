# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
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
    df.drop(columnInfo['target'])
    
    inputDiscrete = df[columnInfo['discrete']]
    inputContinuous = df[columnInfo['continous']]
    
    inputDiscrete = inputDiscrete.as_matrix()
    inputContinuous = processDiscDataframe(inputContinuous)
    
    inputData = np.concatenate((inputDiscrete, inputContinuous), axis=1)
    return inputData, outputData

def file2Dataframe(fileName):
    df = pd.read_csv(fileName, engine='c')
    df = processDataframeForNP(df)
    return df
    
def dataframe2Array(df):
    pass

def saveDataframeAsCSV(df):
    pass

def saveDataframeAsArray(df):
    pass

def seperateTestInputOutput(df):
    pass