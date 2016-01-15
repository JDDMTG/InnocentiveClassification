# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML
from modelsInfo import models

ML.generateTestOutput(dm.load('adaboost_decision_tree.mod'), dm.load(ci.outputDataDirectory + ci.testingInputPathFixed), "OutputData.csv")
