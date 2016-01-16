from dataMngr import save, load
import compInfo as ci
import machineLearning as ML
from modelsInfo import models

def createAndSaveModel(modelDict, indata, outdata, outDict = ''):
    model = ML.generateModel(modelDict['model'], modelDict['modelArrParameters'],
                          modelDict['modelDictParameters'], indata, outdata)
    save(outDict + modelDict['savedModelFileName'], model)
    return model


indata = load(ci.dataDirectory + ci.inputDataPath)
outdata = load(ci.dataDirectory + ci.outputDataPath)
testingdata = load(ci.dataDirectory + ci.testingInputPathFixed)


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

mArr = ['MultilayerPerceptronStandard', 'MultilayerPerceptron10050', 
         'MultilayerPerceptron20050', 'MultilayerPerceptron200100']
modelArr = []

for modelDict in mArr:
    m = models[modelDict]
    model = createAndSaveModel( m, indata, outdata, 
                   ci.modelsDirectory)
    ML.generateTestOutput(model, testingdata, ci.outputDataDirectory + 'outtesting_' + m['modelName'] + '.csv')