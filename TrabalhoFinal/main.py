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
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import roc_curve
    
    from sklearn.model_selection import KFold

    from sklearn.feature_selection import SelectFromModel
    from sklearn.feature_selection import SelectKBest, SelectPercentile
    from sklearn.feature_selection import chi2, f_regression, f_classif
    from sklearn.feature_selection import GenericUnivariateSelect, SelectFwe, SelectFdr, SelectFpr

    from sklearn.ensemble import ExtraTreesClassifier
    
    from pickle import load
    from pickle import dump
    import os

    import matplotlib
    import matplotlib.pyplot as plt

except Exception as a:
    print('Import error', a)

class Main():
    def __init__(self, dataset_androzoo, dataset_androDi, test_size, train_size):
        self.dataset_androzoo = dataset_androzoo
        self.dataset_androDi = dataset_androDi
        
        self.modelsPath = 'MlModels/'
        self.FSFile = 'FS.pkl'
        self.imgsPath = 'rocPlots/'
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

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print(" ––––––––––––––––––––––––––––––––––––– ")
            print("| TruePositive ", tp, "| FalsePositive ", fp,"|")
            print("| FalseNevative ", fn, "| TrueNegative ", tn,"|")
            print(" ––––––––––––––––––––––––––––––––––––– ")

            print("F1Score:\t\t", f1_score(y_test, y_pred, average='binary'))
            print("Recall: \t\t", recall_score(y_test, y_pred, average='binary'))
            print("Precision:\t\t", precision_score(y_test, y_pred, average='binary'))

            # print("", mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7]))
            print("Mean Absolute Error\t", mean_absolute_error(y_test, y_pred))

            # https://scikit-learn.org/stable/modules/model_evaluation.html
            # The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class. The best value is 1 and the worst value is 0 when adjusted=False
            # print("Balanced Accuracy:\t", balanced_accuracy_score(y_test, y_pred))
            # print("Average Precision:\t", average_precision_score(y_test, y_pred))

            # The smaller the Brier score, the better, hence the naming with “loss”. Across all items in a set N predictions, the Brier score measures the mean squared difference between (1) the predicted probability assigned to the possible outcomes for item i, and (2) the actual outcome. Therefore, the lower the Brier score is for a set of predictions, the better the predictions are calibrated.
            if(alg != 'One Class SVM' and alg != 'Multilayer Perceptron' and alg != 'Isolation Forest'):
                print("Brier Score:\t\t", brier_score_loss(y_test, y_pred))
                print("ROC AUC:\t\t", roc_auc_score(y_test, y_pred))


            print(" ", flush=True)
        except Exception as a:
            print('main.printData', a)
    
    def saveImg(self,fileName, img):
        img.savefig(self.imgsPath+fileName)
    
    def plotRocCurve(self, model, X_test, y_test, fileName, label):
        f = plt.figure(1)
        probs = model.predict_proba(X_test)
        probs = probs[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, marker='.', label=label)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        self.saveImg(fileName, f)
        if ('5' in label) or ('-' not in fileName):
            plt.close(f)


    def naiveBayes(self, X_train, X_test, y_train, y_test, fileName):
        try:
            gnb = GaussianNB()
            gnb.fit(X_train, y_train)
            dump(gnb, open(self.modelsPath+fileName, 'wb'))
            y_pred = gnb.predict(X_test)
            self.printData(y_test, y_pred, 'Naive Bayes', 'Multi-Class')
        except Exception as a:
            print('main.naiveBayes', a)

    def KNeighbors(self, X_train, X_test, y_train, y_test, fileName):
        try:
            knn = KNeighborsClassifier(n_neighbors=self.kNeighbors)
            knn.fit(X_train, y_train)
            dump(knn, open(self.modelsPath+fileName, 'wb'))
            y_pred = knn.predict(X_test)
            self.printData(y_test, y_pred, 'KNeighbors', 'Multi-Class')
        except Exception as a:
            print('main.kNeighbors', a)

    def randomForest(self, X_train, X_test, y_train, y_test, fileName):
        try:
            rf = RandomForestClassifier(n_estimators=self.nEstimators)
            rf.fit(X_train, y_train)
            dump(rf, open(self.modelsPath+fileName, 'wb'))
            y_pred = rf.predict(X_test)
            self.printData(y_test, y_pred, 'Random Forest', 'Multi-Class')
        except Exception as a:
            print('main.randomForest', a)

    def adaBoost(self, X_train, X_test, y_train, y_test, fileName):
        try:
            ab = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=self.nEstimators))
            ab.fit(X_train, y_train)
            dump(ab, open(self.modelsPath+fileName, 'wb'))
            y_pred = ab.predict(X_test)
            self.printData(y_test, y_pred, 'Ada Boost', 'Multi-Class')
        except Exception as a:
            print('main.main', a)

    def linearSVC(self, X_train, X_test, y_train, y_test, fileName):
        try:
            lsvc = SVC(gamma='auto', probability=True)
            lsvc.fit(X_train, y_train)
            dump(lsvc, open(self.modelsPath+fileName, 'wb'))
            y_pred = lsvc.predict(X_test)
            self.printData(y_test, y_pred, 'Linear SVC', 'Multi-Class')
        except Exception as a:
            print('main.linearSVC', a)

    def multilayerPerceptron(self, X_train, X_test, y_train, y_test, fileName):
        try:
            mlp = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
            dump(mlp, open(self.modelsPath+fileName, 'wb'))
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
            fs = SelectPercentile().fit(X, y)
            dump(fs, open(self.FSFile, 'wb'))
            X_new = fs.transform(X)
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
        self.mlAlgos(X_train, X_test, y_train, y_test)
    
    def generateModelName(self, model, isKFold, nFold):
        if isKFold:
            return model+' - Fold: '+str(nFold)+'.pkl'
        else:
            return model+'.pkl'
    
    def generateImgName(self, algoName, isKFold):
        algoName = algoName.split('-')[0]
        if isKFold:
            return algoName+' - kFold.png'
        else:
            return algoName+'.png'
    def generateLabel(self, algoName, isKFold, nFold):
        if isKFold:
            return algoName+' - Fold: '+str(nFold)
        else:
            return algoName
    def mlAlgos(self, X_train, X_test, y_train, y_test, isKFold=False, nFold=0):    

        # Naive
        self.naiveBayes(X_train, X_test, y_train, y_test, self.generateModelName('Naive Bayes', isKFold, nFold))
        # KNN
        self.KNeighbors(X_train, X_test, y_train, y_test, self.generateModelName('KNeighbors', isKFold, nFold))
        # Random Forest
        self.randomForest(X_train, X_test, y_train, y_test, self.generateModelName('RF', isKFold, nFold))
        # Ada Boost
        self.adaBoost(X_train, X_test, y_train, y_test, self.generateModelName('Ada Boost', isKFold, nFold))
        # Linear SVC
        self.linearSVC(X_train, X_test, y_train, y_test, self.generateModelName('Linear SVC', isKFold, nFold))
        # MLP
        self.multilayerPerceptron(X_train, X_test, y_train, y_test, self.generateModelName('MLP', isKFold, nFold))
    
    def kFoldMl(self, features, labels):
        print('####### kFold ############# \n ')
        ## Generate 5 folds for train
        kf = KFold(n_splits=5, shuffle=True, random_state =self.randomState)
        i = 1
        for train_index, test_index in kf.split(features, labels):
            print('Fold '+str(i)+':\n')
            ## Make split based on folds
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            ## ML
            self.mlAlgos(X_train, X_test, y_train, y_test, True, i)
            i+=1
    
    def evaluateModels(self, X_test, y_test):
        print('\n ####### Testing Models ############# \n ')
        ## Load and transform data with FS model
        fs = load(open(self.FSFile, 'rb'))
        fs.transform(X_test)

        ## Load models and test with test data
        for model in sorted(os.listdir(self.modelsPath)):
            ml = load(open(self.modelsPath+model, 'rb'))
            y_pred = ml.predict(X_test)
            imgName = ''
            if 'Fold' in model:
                imgName = 'Teste_'+model.split(':')[0]+'.png'
            else:
                imgName = 'Teste_'+model[0:len(model)-4]
            self.printData(y_test, y_pred, model[0:len(model)-4], '')
    def getIndex(self, model, testSplit):
        if '1' in model:
            return testSplit[0]
        if '2' in model:
            return testSplit[1]
        if '3' in model:
            return testSplit[2]
        if '4' in model:
            return testSplit[3]
        if '5' in model:
            return testSplit[4]

    def generateImgs(self, X_test, y_test, isTest):
        path = os.listdir(self.modelsPath)
        notKFold = [ml for ml in path if '-' not in ml]
        kFold = [ml for ml in path if '-' in ml]
        kf = KFold(n_splits=5, shuffle=True, random_state =self.randomState)
        testSplit = []
        testStr = ''
        if isTest:
            testStr = 'Test_'
        for train_index, test_index in kf.split(X_test, y_test):
            testSplit.append(test_index)
        for model in notKFold:
            ml = load(open(self.modelsPath+model, 'rb'))
            self.plotRocCurve(ml, np.array(X_test), np.array(y_test), self.generateImgName(testStr+model[0:len(model)-4], False), self.generateLabel(model[0:len(model)-4], False, 0))
        for model in sorted(kFold):
            ml = load(open(self.modelsPath+model, 'rb'))
            self.plotRocCurve(ml, np.array(X_test.iloc[self.getIndex(model, testSplit)]), np.array(y_test[self.getIndex(model, testSplit)]), self.generateImgName(testStr+model[0:len(model)-4], True), self.generateLabel(model[0:len(model)-4], True, model.split('.')[0].split(': ')[1]))
    
    def main(self):
        ## Read and build dataset with fildered characteristics
        X, y = self.getData()
        
        ## Split data in 8/2 proportion
        X_train, X_test, y_train, y_test = self.featuresLabels(X, y)
        
        ## Feature selection
        # X = self.L1BasedFS(X, y)
        # X = self.SelectKBestFS(X, y, int(X.shape[1]/2), 'chi2')
        X = self.SelectPercentileFS(X_train, y_train)
        # X = self.GenericUnivariateSelectFS(X, y)
        # X = self.TreeBasedFS(X, y, self.nEstimatorsFS)

        ## Normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = pd.DataFrame(min_max_scaler.fit_transform(X_train))
        ## Machine Learning Time
        self.ml(X_train, y_train)
        self.kFoldMl(X_train, y_train)
        ## Evaluation Time
        self.evaluateModels(X_test, y_test)
        self.generateImgs(X_train, y_train, False)
        self.generateImgs(X_test, y_test, True)

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
