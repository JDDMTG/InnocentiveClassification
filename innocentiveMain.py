# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML
from modelsInfo import models
#ML.generateTestOutput(dm.load('RF.np'), dm.load(ci.outputDataDirectory + ci.testingInputPathFixed), "OutputData.csv")

indata = dm.load(ci.outputDataDirectory + ci.inputDataPath)
outdata = dm.load(ci.outputDataDirectory + ci.outputDataPath)

rf200 = models['RandomForest200']
rf = ML.generateModel(rf200['model'], rf200['modelArrParameters'],
                 rf200['modelDictParameters'], indata, outdata)
dm.save(ci.outputDataDirectory + rf200['savedModelFileName'], rf)

naiveBayes = models['NaiveBayesGaussian']
nb = ML.generateModel(naiveBayes['model'], naiveBayes['modelArrParameters'],
                 naiveBayes['modelDictParameters'], indata, outdata)
                 
dm.save(ci.outputDataDirectory + naiveBayes['savedModelFileName'], nb)

