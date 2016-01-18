# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import MultinomialNB
#from sknn.mlp import Classifier, Layer
from sklearn.ensemble import AdaBoostClassifier

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
    'RandomForest200Entropy':{
        'modelName':'rForest_n_200_Ent',
        'modelArrParameters': None,
        'savedModelFileName':'rForest_n_200Ent.mod',
        'modelDictParameters': {
            'criterion': 'entropy',
            'n_estimators':200,
            'n_jobs': -1
        },
        'model': RandomForestClassifier,
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
    },
    'AdaboostDecisionTree':{
        'modelName': 'adaboostDT',
        'modelArrParameters': [DecisionTreeClassifier(max_depth=5)],
        'modelDictParameters': {
            'n_estimators': 200
        }, 
        'model': AdaBoostClassifier,
        'savedModelFileName':'adaboost_decision_tree.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None, 
        
    },
    'AdaboostRF': {
        'modelName': 'adaboostRF',
        'modelArrParameters': [RandomForestClassifier(n_estimators=9, max_depth=5)],
        'modelDictParameters': {
            'n_estimators':25
        },
        'model': AdaBoostClassifier,
        'savedModelFileName': 'adaboost_random_forest.mod'
    }
}

"""
    'MultilayerPerceptronStandard':{
        'modelName':'mlpStandard',
        'modelArrParameters': None,
        'modelDictParameters': {
            'layers': [
                Layer("Rectifier", units=200),
                Layer("Softmax")]
        }, 
        'model': Classifier,
        'savedModelFileName':'mlpStandard.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,     
    },
    'MultilayerPerceptron10050':{
        'modelName':'mlp10050',
        'modelArrParameters': None,
        'modelDictParameters': {
            'layers': [
                Layer("Rectifier", units=100),
                Layer("Tanh", units=100),
                Layer("Softmax")]
        }, 
        'model': Classifier,
        'savedModelFileName':'mlp10050.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,     
    },
    'MultilayerPerceptron20050':{
        'modelName':'mlp20050',
        'modelArrParameters': None,
        'modelDictParameters': {
            'layers': [
                Layer("Rectifier", units=200),
                Layer("Tanh", units=50),
                Layer("Softmax")],
            'n_iter': 3
        }, 
        'model': Classifier,
        'savedModelFileName':'mlp20050.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,     
    },
    'MultilayerPerceptron200100':{
        'modelName':'mlp10050',
        'modelArrParameters': None,
        'modelDictParameters': {
            'layers': [
                Layer("Rectifier", units=200),
                Layer("Tanh", units=100),
                Layer("Softmax")]
        }, 
        'model': Classifier,
        'savedModelFileName':'mlp200100.mod',
        'percentYesCorrect': None,
        'percentNoCorrect': None, 
        'numInstances': None,         
    },

"""