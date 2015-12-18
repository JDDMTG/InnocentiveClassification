# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier

models = {
    RandomForest:{
        'modelName':'rForest_n_10',
        'modelParameters': {
            'n':10
            }, 
        'model': RandomForest,
        'savedModelFileName':'rForest_n_10.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
            }
        }