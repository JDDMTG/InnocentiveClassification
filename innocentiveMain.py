# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML
from modelsInfo import models
#ML.generateTestOutput(dm.load('RF.np'), dm.load(ci.outputDataDirectory + ci.testingInputPathFixed), "OutputData.csv")

ML.generateTestOutput(dm.load('adaboost_decision_tree.mod'), dm.load(ci.outputDataDirectory + ci.testingInputPathFixed), "OutputData.csv")

def createAndSaveModel(modelDict, indata, outdata, outDict = ''):
    model = ML.generateModel(modelDict['model'], modelDict['modelArrParameters'],
                          modelDict['modelDictParameters'], indata, outdata)
    dm.save(outDict + model['savedModelFileName'])
    return model


indata = dm.load(ci.outputDataDirectory + ci.inputDataPath)
outdata = dm.load(ci.outputDataDirectory + ci.outputDataPath)

createAndSaveModel(models['RandomForest200'], indata, outdata, 
                   ci.outputDataDirectory)
                   
createAndSaveModel(models['NaiveBayesGaussian'], indata, outdata, 
                   ci.outputDataDirectory)
                   