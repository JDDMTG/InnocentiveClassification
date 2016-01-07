# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import dataMngr as dm
import compInfo as ci

# only generated model, doesn't attempt to fit model
def generateModel(model, modelArrParams, modelDictParams, inData, outData):
    if modelArrParams is None:
        modelArrParams = []
    if modelDictParams is None:
        modelDictParams = {}
    standardModel = model(*modelArrParams, **modelDictParams)
    return standardModel.fit(inData, outData.ravel())

def percentCorrect(modelData, outputData):
    labeledCorrectlyCount = 0
    labeledIncorrectlyCount = 0
    for i in xrange(len(modelData)):
        if modelData[i] == outputData[i]:
            labeledCorrectlyCount += 1
        else:
            labeledIncorrectlyCount += 1
    return float(labeledCorrectlyCount)/len(modelData)

def generateModelScores(generatedModel, inData, outData, modelFileName, statisticsFileName):
  predicted = generatedModel.predict(inData)
  pc = percentCorrect(predicted, outData)
  saveModelAndScores(modelFileName, generatedModel, statisticsFileName, pc)
  return pc

def saveModelAndScores(modelFileName, generatedModel, statisticsFileName, scores):
  dm.save(modelFileName, generatedModel)
  dm.save(statisticsFileName, scores)

def modelDriver(model, modelArrParams, modelDictParams, inData, outData):
  generatedModel = generateModel(model, modelArrParams, modelDictParams, inData, outData)
  pc = generateModelScores(generatedModel, inData, outData, "RF.np", "RF_Stats.np")
  print pc

def generateTestOutput(generatedModel, inData, outDataName):
  outData = generatedModel.predict(inData)
  dm.writeToFilePath(outData, outDataName)

def makeEditedTestOutput(generatedModel, inData, outDataName):
  inData[425936][6] = 0
  inData[617750][6] = 0
  inData[734618][6] = 0
  generatedModel(generatedModel, inData, outDataName)

# modelDriver(RandomForestClassifier, None, None, dm.load('Innocentive_500_Sample_input.np'), dm.load('Innocentive_500_Sample_output.np'))
# modelDriver(RandomForestClassifier, None, None, dm.load(ci.outputDataDirectory + ci.inputDataPath), dm.load(ci.outputDataDirectory + ci.outputDataPath))
# generateTestOutput(dm.load('RF.np'), ci.outputDataDirectory + ci.testingInputPath, ci.outputDataDirectory + ci.)
# generateTestOutput(dm.load('RF.np'), dm.load(ci.outputDataDirectory + ci.testingInputPath), "OutputData.csv")
# print dm.load(ci.outputDataDirectory + ci.testingInputPath)
makeEditedTestOutput(dm.load('RF.np'), dm.load(ci.outputDataDirectory + ci.testingDF), "OutputData.csv")


