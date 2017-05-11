#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn import tree
import pydotplus

import pandas as pd
import numpy

def principal_components_analysis(data_without_target, target, n_components):
    X = data_without_target
    Y = target

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Model declaration
    if n_components < 1:
        pca = PCA(n_components = n_components, svd_solver = 'full')
    else:
        pca = PCA(n_components = n_components)

    # Model training
    pca.fit(X)

    # Model transformation
    new_feature_vector = pca.transform(X)

    # Model information:
    print('\nModel information:\n')
    print('Number of components elected: ' + str(pca.n_components))
    print('New feature dimension: ' + str(pca.n_components_))
    print('Variance of every feature: ' + str(pca.explained_variance_ratio_))

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])
    return new_feature_vector


def convert_data_to_numeric(data):
    numpy_data = data.values

    for i in range(len(numpy_data[0])):
        temp = numpy_data[:,i]
        dict = numpy.unique(numpy_data[:,i])
        print("---------------------------------------------")
        print(i)
        print(dict)
        print("---------------------------------------------")
        if type(dict[0]) == str:
            for j in range(len(dict)):
                temp[numpy.where(numpy_data[:,i] == dict[j])] = j
            numpy_data[:,i] = temp
    return numpy_data

def z_score_normalization(data_without_target, target):
    X = data_without_target
    Y = target

    # First 10 rows
    print('Training Data:\n\n' + str(X[:10]))
    print('\n')
    print('Targets:\n\n' + str(Y[:10]))

    # Data standarization
    standardized_data = preprocessing.scale(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(standardized_data[:10])
    print("----------------------------------------------- FINALIZA NORMALIZACION -------------------------")
    return standardized_data

def min_max_scaler(data_without_target, target):
    X = data_without_target
    Y = target
    # Data normalization
    min_max_scaler = preprocessing.MinMaxScaler()

    min_max_scaler.fit(X)

    # Model information:
    print('\nModel information:\n')
    print('Data min: ' + str(min_max_scaler.data_min_))
    print('Data max: ' + str(min_max_scaler.data_max_))

    new_feature_vector = min_max_scaler.transform(X)

    # First 10 rows of new feature vector
    print('\nNew feature vector:\n')
    print(new_feature_vector[:10])
    print("---------------------- TERMINA NORMALIZACION")
    return new_feature_vector

def replace_missing_values_with_constant(data, column, constant):
    temp = data[column].fillna(constant)
    data[column] = temp
    return data

def show_data_info(data):
    print("Number of instance: " + str(data.shape[0])); #shape da la forma de la medida de los atributos y las instancias [0] -> numero de instancias
    print("Number of fetures: " + str(
        data.shape[1]));  # shape da la forma de la medida de los atributos y las instancias [1] -> numero de campos
    print("----------------------------------------------------------------")
    print(data.head(10)) #Te muestra los titulos de cada una de las caracteristicas y un número de instancias
    print("Atributos numericos:")
    numerical_info = data.iloc[:, : data.shape[1]]
    print(numerical_info.describe())

    # """
    # This function returns four subsets that represents training and test data
    # :param data: numpy array
    # :return: four subsets that represents data train and data test
    # """
def data_splitting(data_features, data_targets, test_size):
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        train_test_split(data_features,
                         data_targets,
                         test_size = test_size, random_state=10)
    return data_features_train, data_features_test, data_targets_train, data_targets_test

def decision_tree_training(data_without_target, target):
    data_features = data_without_target
    data_targets = target

    #Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = \
        data_splitting(data_features, data_targets, 0.25)

    #Model declaration
    """
    Parameters to select:
    criterion: "entropy" or "gini": default: gini
    max_depth: maximum depth of tree, default: None
    """
    dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
    dec_tree.fit(data_features_train, data_targets_train)
    #Model evaluation
    test_data_predicted = dec_tree.predict(data_features_test)
    error = metrics.mean_absolute_error(data_targets_test, test_data_predicted)
    print("Model error: " + str(error))
    # print("Probability of each class: \n")
    #Measure probability of each class
    # prob_class = dec_tree.predict_proba(data_features_test)
    # print(prob_class)
    # print("Feature Importance: \n")
    # print(dec_tree.feature_importances_)

if __name__ == '__main__':

    # PRIMERA ITERACIÓN
    # CSV que salio de la limpieza de datos que hice en mongo
    # data = pd.read_csv('cleanReadyForPca.csv')

    # TUVE QUE SEPARAR EL TARGET DEL DATASET PARA LOS ALGORITMOS Y LA NORMALIZACIÖN
    # target = data['SalePrice']
    # data_without_target = data.drop("SalePrice", 1)

    # LO QUE HICE PARA SUSITTUIR LOS NAN QUE SALIAN
    # data_without_target = replace_missing_values_with_constant(data, "Alley", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "MasVnrType", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtQual", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtCond", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtExposure", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtFinType1", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtFinType2", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "Electrical", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "MiscFeature", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "FireplaceQu", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageYrBlt", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageQual", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageFinish", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "GarageCond", "NA")
    #
    # data_only_numeric = convert_data_to_numeric(data_without_target)
    #
    # data_normalizado = z_score_normalization(data_only_numeric, target)
    # data_after_pca = principal_components_analysis(data_normalizado, target, .90)
    #
    # decision_tree_training(data_after_pca, target)

    # Segunda iteración
    # CSV que salio de la limpieza de datos que hice en mongo
    # data = pd.read_csv('cleanReadyForPcaSegundaIteracion.csv')
    #
    # # TUVE QUE SEPARAR EL TARGET DEL DATASET PARA LOS ALGORITMOS Y LA NORMALIZACIÖN
    # target = data['SalePrice']
    # data_without_target = data.drop("SalePrice", 1)
    #
    # data_without_target = replace_missing_values_with_constant(data, "Alley", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtExposure", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "MiscFeature", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "PoolQC", "NA")
    # data_without_target = replace_missing_values_with_constant(data_without_target, "MasVnrArea", -1)
    #
    # show_data_info(data_without_target)
    # data_only_numeric = convert_data_to_numeric(data_without_target)
    # data_normalizado = min_max_scaler(data_only_numeric, target)
    # #NO SE UTILIZO PCA POR QUE BORRE MANUALMENTE LAS COLUMNAS EN MONGO
    # # data_after_pca = principal_components_analysis(data_normalizado, target, .90)
    # decision_tree_training(data_normalizado, target)

    # Tercera iteración
    # CSV que salio de la limpieza de datos que hice en mongo
    data = pd.read_csv('cleanReadyForPcaSegundaIteracion.csv')

    # TUVE QUE SEPARAR EL TARGET DEL DATASET PARA LOS ALGORITMOS Y LA NORMALIZACIÖN
    target = data['SalePrice']
    data_without_target = data.drop("SalePrice", 1)

    data_without_target = replace_missing_values_with_constant(data, "Alley", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "BsmtExposure", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "MiscFeature", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "PoolQC", "NA")
    data_without_target = replace_missing_values_with_constant(data_without_target, "MasVnrArea", -1)

    show_data_info(data_without_target)
    data_only_numeric = convert_data_to_numeric(data_without_target)
    data_normalizado = z_score_normalization(data_only_numeric, target)
    #NO SE UTILIZO PCA POR QUE BORRE MANUALMENTE LAS COLUMNAS EN MONGO
    decision_tree_training(data_normalizado, target)
