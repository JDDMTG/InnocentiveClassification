# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo
import machineLearning as ML

ML.generateTestOutput(dm.load('RF.np'), dm.load(ci.outputDataDirectory + ci.testingInputPathFixed), "OutputData.csv")

