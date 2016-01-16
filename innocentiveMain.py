# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML

ML.generateTestOutput(dm.load(ci.modelsDirectory + 'rForest_n_200Ent.mod'), dm.load(ci.dataDirectory + ci.testingInputPathFixed), ci.outputDataDirectory + "OutputDataEntropy200.csv")