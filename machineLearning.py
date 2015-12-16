# -*- coding: utf-8 -*-

def generateModel(inFilePath, outFilePath):
    inData = file2Data(inFilePath)
    outData = file2Data(outFilePath)
    print len(outData)
    print outData
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(inData, outData)