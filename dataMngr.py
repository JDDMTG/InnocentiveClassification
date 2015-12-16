# -*- coding: utf-8 -*-
import pandas as pd
import csv
from dataInfo import columnInfo

def processDataframeForNP(df):
    df = df.drop(columnInfo['extraneous'], axis=1) #Unecessary Info
    pass

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

def dataToInputOutput(fileName):
  with open(fileName, 'wb') as csvfile:
    linelist = [line for line in csvfile.readlines()]
    print linelist

dataToInputOutput('../InnoCentive_Challenge_9933493_training_data.csv')
