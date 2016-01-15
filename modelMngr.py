import compInfo as ci
import machineLearning as ML
from modelsInfo import models
import cpickle as pickle

def save(fileName, obj):
    pickle.dump( obj, open( fileName, "wb" ) )
    pass

def load(fileName):
    return pickle.load( open( fileName, "rb" ) )


def createAndSaveModel(modelDict, indata, outdata, outDict = ''):
    model = ML.generateModel(modelDict['model'], modelDict['modelArrParameters'],
                          modelDict['modelDictParameters'], indata, outdata)
    save(outDict + modelDict['savedModelFileName'], model)
    return model


indata = load(ci.outputDataDirectory + ci.inputDataPath)
outdata = load(ci.outputDataDirectory + ci.outputDataPath)


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

model = createAndSaveModel(models['RandomForest200Entropy'], indata, outdata, 
                   ci.outputDataDirectory)