# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

models = {
    'RandomForest200':{
        'modelName':'rForest_n_200',
        'modelArrParameters': None,
        'modelDictParameters': {
            'n_estimators':200,
            'n_jobs': -1
            }, 
        'model': RandomForestClassifier,
        'savedModelFileName':'rForest_n_200.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
            },
    'NaiveBayesGaussian':{
        'modelName':'nBayes',
        'modelArrParameters': None,
        'modelDictParameters': None, 
        'model': GaussianNB,
        'savedModelFileName':'naive_bayes.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
            }
        }