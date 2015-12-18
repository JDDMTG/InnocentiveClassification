# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import dataMngr as dm

# only generated model, doesn't attempt to fit model
def generateModel(model, modelArrParams, modelDictParams):
    if modelArrParams is None:
        modelArrParams = []
    if modelDictParams is None:
        modelDictParams = {}
    standardModel = model(*modelArrParams, **modelDictParams)
    return standardModel

def generateModelScores(stdModel, inData, outData, modelFileName, statisticsFileName):
  outData = outData.ravel()
  scores = cross_val_score(stdModel, inData, outData)
  saveModelAndScores(modelFileName, stdModel, statisticsFileName, scores)
  return scores.mean()

def saveModelAndScores(modelFileName, stdModel, statisticsFileName, scores):
  dm.save(modelFileName, stdModel)
  dm.save(statisticsFileName, scores.mean())

def generateModelAndScores(model, modelArrParams, modelDictParams, inData, outData, numRuns):
  for x in xrange(numRuns):
    stdModel = generateModel(model, modelArrParams, modelDictParams)
    print generateModelScores(stdModel, inData, outData, "RF" + str(x) + ".np", "RF_Stats" + str(x) + ".np")


print generateModelAndScores(RandomForestClassifier, None, None, dm.load("Innocentive_500_Sample_input.np"), dm.load("Innocentive_500_Sample_output.np"), 5)