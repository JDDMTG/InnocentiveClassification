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

#mArr = ['MultilayerPerceptronStandard', 'MultilayerPerceptron10050', 
#         'MultilayerPerceptron20050', 'MultilayerPerceptron200100']

mArr = ['AdaboostRFd_4_t_9_f_25', 'AdaboostRFd_6_t_9_f_25', 'AdaboostRFd_7_t_9_f_25']
modelArr = []

for modelDict in mArr:
    m = models[modelDict]
    model = createAndSaveModel( m, indata, outdata, 
                   ci.modelsDirectory)
    modelArr.append(model)

testingdata = load(ci.dataDirectory + ci.testingInputPathFixed)

for i in xrange(len(mArr)):
    m = models[mArr[i]]
    model = modelArr[i]
    ML.generateTestOutput(model, testingdata, ci.outputTestingDirectory + m['savedModelFileName'] + '.csv')