import dataMngr as dm
import compInfo as ci
import machineLearning as ML
from modelsInfo import models

def createAndSaveModel(modelDict, indata, outdata, outDict = ''):
    model = ML.generateModel(modelDict['model'], modelDict['modelArrParameters'],
                          modelDict['modelDictParameters'], indata, outdata)
    dm.save(outDict + modelDict['savedModelFileName'], model)
    return model


indata = dm.load(ci.outputDataDirectory + ci.inputDataPath)
outdata = dm.load(ci.outputDataDirectory + ci.outputDataPath)


"""
# createAndSaveModel(models['RandomForest200'], indata, outdata, 
#                   ci.outputDataDirectory)
                   
# createAndSaveModel(models['NaiveBayesGaussian'], indata, outdata, 
#                   ci.outputDataDirectory)

multilayerPerceptrons = ['MultilayerPerceptronStandard', 'MultilayerPerceptron10050',
       'MultilayerPerceptron20050', 'MultilayerPerceptron200100']
       
for mlp in multilayerPerceptrons:
    createAndSaveModel(models[mlp], indata, outdata, ci.outputDataDirectory)
    
"""

model = createAndSaveModel(models['AdaboostDecisionTree'], indata, outdata, 
                   ci.outputDataDirectory)