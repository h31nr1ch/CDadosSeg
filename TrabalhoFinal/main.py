#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tiago Heinrich & Rodrigo Lemos

try:
    import argparse
    import sys

    import numpy as np
    import pandas as pd

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import brier_score_loss
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import jaccard_score

    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import SelectKBest, SelectPercentile
    from sklearn.feature_selection import chi2, f_regression, f_classif
    from sklearn.feature_selection import GenericUnivariateSelect, SelectFwe, SelectFdr, SelectFpr

    from sklearn.ensemble import ExtraTreesClassifier

    import matplotlib
    import matplotlib.pyplot as plt

except Exception as a:
    print('Import error', a)

class Main():
    def __init__(self, dataset_androzoo, dataset_androDi, test_size, train_size):
        self.dataset_androzoo = dataset_androzoo
        self.dataset_androDi = dataset_androDi

        self.testSize = test_size
        self.trainSize = train_size
        self.randomState = 42

        self.kNeighbors = 3
        self.nEstimators = 100
        self.nEstimatorsFS = 100

    def convertString(self, stringInput):
        try:
            return stringInput.replace('\'','').replace('[','').replace(']','')
        except Exception as a:
            print('main.convertString:', a)

    def crazyMarkVector(self, dfm, vector, mark):
        try:
            ll=[]
            for each in mark:
                if(each in vector):
                    ll.append(1)
                else:
                    ll.append(0)
            dfm.loc[dfm.shape[0] + 1] = ll
            return  dfm
        except Exception as a:
            print('main.crazyMarkVector:', a)

    def getData(self):
        try:
            # Read data Androzoo (Fabricio)
            dfv1 = pd.read_csv(self.dataset_androzoo, header = 0, delimiter = ",")

            # Read data AndroDi (Rodrigo masters dataset)
            dfv2 = pd.read_csv(self.dataset_androDi, header = 0, delimiter = ",")

            # print(dfv2.columns.values.tolist())
            # print(dfv1.columns.values.tolist())

            dataset = pd.merge(dfv1, dfv2, left_on='meta.pkg.name', right_on='Apk')
            # print(dataset['Apk'].iloc[0], dataset['meta.pkg.name'].iloc[0])

            ## Remove Nan
            dataset.dropna(axis=1, how='all')
            dataset.fillna(0)

            ## Drop and copy label
            y = np.array(dataset['Label'])
            del dataset['Label']

            ## Build new characteristics columns for the dataset
            # for each in ['resource.entry', 'manifest.permission',  'manifest.category']:
            rr = []
            for each in ['manifest.permission',  'manifest.category', 'source.class.package']:
                ll = []
                for elem in list(dataset[each]):
                    for x in elem.split(" "):
                        if(type(x) == list):
                            for eachX in x:
                                ll.append(self.convertString( (eachX.split('.')[-1:])[0] ) )
                        else:
                            ll.append( self.convertString( (x.split('.')[-1:])[0] ) )
                # print(each, len(set(ll)))
                rr.append(set(ll))

            aux = rr[0].union(rr[1])
            rr  = aux.union(rr[2])

            ## Get just One (test each list column from the Fab. dataset)
            # rr = []
            # for each in dataset['manifest.permission'].values:
            #     if (type(each) == list):
            #         for elemEach in each:
            #             rr.append(self.convertString( (elemEach.split('.')[-1:])[0] ))
            #     else:
            #         rr.append(self.convertString( (each.split('.')[-1:])[0] ))
            # rr = (set(rr))

            ## Get dataset second part
            dfSecond = pd.DataFrame(columns=list(rr))

            for index, row in dataset.iterrows():
                ll = []

                xX = (dataset['manifest.category'].iloc[index])
                xY = (dataset['manifest.permission'].iloc[index])
                xZ = (dataset['source.class.package'].iloc[index])

                for eachxX in xX.split(" "):
                    if(type(eachxX) == list):
                        for seilaToSemNomeX in eachxX:
                            ll.append(self.convertString( (seilaToSemNomeX.split('.')[-1:])[0] ) )
                    else:
                        ll.append( self.convertString( (eachxX.split('.')[-1:])[0] ) )

                for eachxY in xY.split(" "):
                    if(type(eachxY) == list):
                        for seilaToSemNomeY in eachxY:
                            ll.append(self.convertString( (seilaToSemNomeY.split('.')[-1:])[0] ) )
                    else:
                        ll.append( self.convertString( (eachxY.split('.')[-1:])[0] ) )


                for eachxY in xZ.split(" "):
                    if(type(eachxY) == list):
                        for seilaToSemNomeY in eachxY:
                            ll.append(self.convertString( (seilaToSemNomeY.split('.')[-1:])[0] ) )
                    else:
                        ll.append( self.convertString( (eachxY.split('.')[-1:])[0] ) )

                dfSecond = self.crazyMarkVector(dfSecond, ll, rr)

            ## Get data with some value
            # print(dataset.columns.values.tolist())

            dataset = dataset[['meta.dex.size', 'manifest.tarsdk', 'manifest.minsdk', 'manifest.maxsdk', 'BC_REPLY_SG', 'BC_TRANSACTION', 'BC_REPLY', 'BC_ACQUIRE_RESULT', 'BC_FREE_BUFFER', 'BC_INCREFS', 'BC_ACQUIRE', 'BC_RELEASE', 'BC_DECREFS', 'BC_INCREFS_DONE', 'BC_ACQUIRE_DONE', 'BC_ATTEMPT_ACQUIRE', 'BC_REGISTER_LOOPER', 'BC_ENTER_LOOPER', 'BC_EXIT_LOOPER', 'BC_REQUEST_DEATH_NOTIFICATION', 'BC_CLEAR_DEATH_NOTIFICATION', 'BC_DEAD_BINDER_DONE', 'BC_TRANSACTION_SG', 'BR_ERROR', 'BR_OK', 'BR_TRANSACTION', 'BR_ACQUIRE_RESULT', 'BR_DEAD_REPLY', 'BR_TRANSACTION_COMPLETE', 'BR_INCREFS', 'BR_ACQUIRE', 'BR_RELEASE', 'BR_DECREFS', 'BR_ATTEMPT_ACQUIRE', 'BR_NOOP', 'BR_SPAWN_LOOPER', 'BR_FINISHED', 'BR_DEAD_BINDER', 'BR_CLEAR_DEATH_NOTIFICATION_DONE', 'BR_FAILED_REPLY', 'BR_REPLY']]

            # print(dfSecond.join(dataset))
            dataset = dfSecond.join(dataset) # Merge the two dataFrames

            return dataset.fillna(0), y
        except Exception as a:
            print('main.getData:', a)

    def printData(self, y_test, y_pred, alg, type):
        try:
            print("Algorithm:\t\t", alg, ' (', type,')')
            print("F1Score:\t\t", f1_score(y_test, y_pred, average='binary'))
            print("Recall: \t\t", recall_score(y_test, y_pred, average='binary'))
            print("Precision:\t\t", precision_score(y_test, y_pred, average='binary'))

            # https://scikit-learn.org/stable/modules/model_evaluation.html
            # The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class. The best value is 1 and the worst value is 0 when adjusted=False
            print("Balanced Accuracy:\t", balanced_accuracy_score(y_test, y_pred))
            print("Average Precision:\t", average_precision_score(y_test, y_pred))
            # The smaller the Brier score, the better, hence the naming with “loss”. Across all items in a set N predictions, the Brier score measures the mean squared difference between (1) the predicted probability assigned to the possible outcomes for item i, and (2) the actual outcome. Therefore, the lower the Brier score is for a set of predictions, the better the predictions are calibrated.
            if(alg != 'One Class SVM' and alg != 'Multilayer Perceptron' and alg != 'Isolation Forest'):
                print("Brier Score:\t\t", brier_score_loss(y_test, y_pred))
                print("ROC AUC:\t\t", roc_auc_score(y_test, y_pred))
            print(" ", flush=True)
        except Exception as a:
            print('main.printData', a)

    def naiveBayes(self, X_train, X_test, y_train, y_test):
        try:
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            y_pred = gnb.predict(X_test)
            self.printData(y_test, y_pred, 'Naive Bayes', 'Multi-Class')
        except Exception as a:
            print('main.naiveBayes', a)

    def KNeighbors(self, X_train, X_test, y_train, y_test):
        try:
            knn = KNeighborsClassifier(n_neighbors=self.kNeighbors)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            self.printData(y_test, y_pred, 'KNeighbors', 'Multi-Class')
        except Exception as a:
            print('main.kNeighbors', a)

    def randomForest(self, X_train, X_test, y_train, y_test):
        try:
            rf = RandomForestClassifier(n_estimators=self.nEstimators)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            self.printData(y_test, y_pred, 'Random Forest', 'Multi-Class')
        except Exception as a:
            print('main.randomForest', a)

    def adaBoost(self, X_train, X_test, y_train, y_test):
        try:
            ab = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=self.nEstimators))
            ab.fit(X_train, y_train)
            y_pred = ab.predict(X_test)
            self.printData(y_test, y_pred, 'Ada Boost', 'Multi-Class')
        except Exception as a:
            print('main.main', a)

    def linearSVC(self, X_train, X_test, y_train, y_test):
        try:
            lsvc = SVC(gamma='auto')
            lsvc.fit(X_train, y_train)
            y_pred = lsvc.predict(X_test)
            self.printData(y_test, y_pred, 'Linear SVC', 'Multi-Class')
        except Exception as a:
            print('main.linearSVC', a)

    def multilayerPerceptron(self, X_train, X_test, y_train, y_test):
        try:
            mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            self.printData(y_test, y_pred, 'Multilayer Perceptron', 'One-Class')
        except Exception as a:
            print('main.multilayerPerceptron', a)

    def printNewShape(self, Old, new):
        try:
            print('Old Shape:' , Old.shape, 'New Shape:', new.shape)
        except Exception as a:
            print('main.printNewShape', a)

    def L1BasedFS(self, X, y):
        try:
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
            model = SelectFromModel(lsvc, prefit=True)
            X_new = model.transform(X)
            self.printNewShape(X, X_new)
            return X_new
        except Exception as a:
            print('main.L1BasedFeatureSelection', a)

    def SelectKBestFS(self, X, y, N_bestFeatures, alg):
        try:
            # SelectKBest removes all but the highest scoring features
            if(alg == 'chi2'):
                X_new = SelectKBest(score_func=chi2, k=N_bestFeatures).fit_transform(X, y)
            elif(alg == 'f_classif'):
                X_new = SelectKBest(score_func=f_classif, k=N_bestFeatures).fit_transform(X, y)
            elif(alg == 'f_regression'):
                X_new = SelectKBest(score_func=f_regression, k=N_bestFeatures).fit_transform(X, y)
            self.printNewShape(X, X_new)
            return X_new
        except Exception as a:
            print('main.SelectKBestFS', a)

    def SelectPercentileFS(self, X, y):
        try:
            X_new = SelectPercentile().fit_transform(X, y)
            self.printNewShape(X, X_new)
            return X_new
        except Exception as a:
            print('main.SelectPercentileFS', a)

    def GenericUnivariateSelectFS(self, X, y):
        try:
            transformer = GenericUnivariateSelect(chi2, mode='fwe' ) # ['chi2', 'mutual_info_classif', 'f_classif', 'f_regression', 'mutual_info_regression'] ['percentile', 'k_best', 'fpr', 'fdr', 'fwe']
            X_new = transformer.fit_transform(X, y)
            self.printNewShape(X, X_new)
            return X_new
        except Exception as a:
            print('main.GenericUnivariateSelectFS', a)

    def TreeBasedFS(self, X, y, n_esti):
        try:
            clf = ExtraTreesClassifier(n_estimators=n_esti)
            clf = clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            X_new = model.transform(X)
            self.printNewShape(X, X_new)
            return X_new
        except Exception as a:
            print('main.TreeBasedFS', a)

    def featuresLabels(self, features, labels):
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=self.testSize, train_size=self.trainSize, random_state=self.randomState)
            return X_train, X_test, y_train, y_test
        except Exception as a:
            print('main.featuresLabels', a)

    def ml(self, features, labels):

        ## Get data for MultiClass
        X_train, X_test, y_train, y_test = self.featuresLabels(features, labels)

        ## Generate plot data distribution

        # Naive
        self.naiveBayes(X_train, X_test, y_train, y_test)
        # KNN
        self.KNeighbors(X_train, X_test, y_train, y_test)
        # Random Forest
        self.randomForest(X_train, X_test, y_train, y_test)
        # Ada Boost
        self.adaBoost(X_train, X_test, y_train, y_test)
        # Linear SVC
        self.linearSVC(X_train, X_test, y_train, y_test)
        # MLP
        self.multilayerPerceptron(X_train, X_test, y_train, y_test)

    def main(self):
        ## Read and build dataset with fildered characteristics
        X, y = self.getData()

        ## Feature selection
        # X = self.L1BasedFS(X, y)
        # X = self.SelectKBestFS(X, y, int(X.shape[1]/2), 'chi2')
        X = self.SelectPercentileFS(X, y)
        # X = self.GenericUnivariateSelectFS(X, y)
        # X = self.TreeBasedFS(X, y, self.nEstimatorsFS)

        ## Normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        X = pd.DataFrame(min_max_scaler.fit_transform(X))

        ## Machine Learning Time
        self.ml(X, y)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CDS Trabalho Final.')

    parser.add_argument('--version','-v','-vvv','-version', action='version', version=str('Base 0.2'))

    parser.add_argument('--dataset-androzoo', type=str, required=True, help='Input dataset androzoo file.')

    parser.add_argument('--dataset-androDi', type=str, required=True, help='Input dataset AndroDi file.')

    parser.add_argument('--test-size', type=int, default=0.2, required=False, help='Test subsets portion (default 0.2).')

    parser.add_argument('--train-size', type=int, default=0.8, required=False, help='Train subsets portion (default 0.8).')

    args = parser.parse_args()
    kwargs = {
        'dataset_androzoo': args.dataset_androzoo,
        'dataset_androDi': args.dataset_androDi,
        'test_size': args.test_size,
        'train_size': args.train_size
    }

    try:
        worker = Main(**kwargs)
        worker.main()

    except KeyboardInterrupt as e:
        print('Exit using ctrl^C')
