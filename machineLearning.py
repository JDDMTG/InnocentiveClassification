# -*- coding: utf-8 -*-


def generateModel(model, modelArrParams, modelDictParams, inData, outData):
    if modelArrParams is None:
        modelArrParams = []
    if modelDictParams is None:
        modelDictParams = {}
    standardModel = model(*modelArrParams, **modelDictParams)
    return standardModel.fit(inData, outData)