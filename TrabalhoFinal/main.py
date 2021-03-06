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

    from sklearn.feature_extraction.text import CountVectorizer
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

except Exception as a:
    print('Import error', a)

class Main():
    def __init__(self, dataset_androzoo, dataset_androDi):
        self.dataset_androzoo = dataset_androzoo
        self.dataset_androDi = dataset_androDi

        self.testSize = 0.5
        self.randomState = 42

        self.kNeighbors = 3
        self.nEstimators = 100

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
            for each in ['manifest.permission',  'manifest.category']:
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

            rr = rr[0].union(rr[1])

            ## Get dataset second part
            dfSecond = pd.DataFrame(columns=list(rr))

            for index, row in dataset.iterrows():
                ll = []

                xX = (dataset['manifest.category'].iloc[index])
                xY = (dataset['manifest.permission'].iloc[index])

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

                dfSecond = self.crazyMarkVector(dfSecond, ll, rr)

            ## Get data with some value
            # print(dataset.columns.values.tolist())

            dataset = dataset[['meta.vt.score','meta.dex.size', 'manifest.tarsdk', 'manifest.minsdk', 'manifest.maxsdk', 'BC_REPLY_SG', 'BC_TRANSACTION', 'BC_REPLY', 'BC_ACQUIRE_RESULT', 'BC_FREE_BUFFER', 'BC_INCREFS', 'BC_ACQUIRE', 'BC_RELEASE', 'BC_DECREFS', 'BC_INCREFS_DONE', 'BC_ACQUIRE_DONE', 'BC_ATTEMPT_ACQUIRE', 'BC_REGISTER_LOOPER', 'BC_ENTER_LOOPER', 'BC_EXIT_LOOPER', 'BC_REQUEST_DEATH_NOTIFICATION', 'BC_CLEAR_DEATH_NOTIFICATION', 'BC_DEAD_BINDER_DONE', 'BC_TRANSACTION_SG', 'BR_ERROR', 'BR_OK', 'BR_TRANSACTION', 'BR_ACQUIRE_RESULT', 'BR_DEAD_REPLY', 'BR_TRANSACTION_COMPLETE', 'BR_INCREFS', 'BR_ACQUIRE', 'BR_RELEASE', 'BR_DECREFS', 'BR_ATTEMPT_ACQUIRE', 'BR_NOOP', 'BR_SPAWN_LOOPER', 'BR_FINISHED', 'BR_DEAD_BINDER', 'BR_CLEAR_DEATH_NOTIFICATION_DONE', 'BR_FAILED_REPLY', 'BR_REPLY']]

            # print(dfSecond.join(dataset))
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

    def featuresLabels(self, features, labels):
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=self.testSize, random_state=self.randomState)
            return X_train, X_test, y_train, y_test
        except Exception as a:
            print('ML.featuresLabels', a)

    def ml(self, features, labels):

        ## Get data for MultiClass
        X_train, X_test, y_train, y_test = self.featuresLabels(features, labels)

        # Naive
        self.naiveBayes(X_train, X_test, y_train, y_test)
        # KNN
        self.KNeighbors(X_train, X_test, y_train, y_test)
        # Random Forest
        self.randomForest(X_train, X_test, y_train, y_test)
        # Outro
        pass

    def main(self):
        # Read and build dataset with fildered characteristics
        X, y = self.getData()

        # # Normalization
        # min_max_scaler = preprocessing.MinMaxScaler()
        # final = pd.DataFrame(min_max_scaler.fit_transform(df))



        # Machine Learning Time
        self.ml(X, y)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CDS Trabalho Final.')

    parser.add_argument('--version','-v','-vvv','-version', action='version', version=str('Base 0.1'))

    parser.add_argument('--dataset-androzoo', type=str, required=True, help='Input dataset androzoo file.')

    parser.add_argument('--dataset-androDi', type=str, required=True, help='Input dataset AndroDi file.')

    args = parser.parse_args()
    kwargs = {
        'dataset_androzoo': args.dataset_androzoo,
        'dataset_androDi': args.dataset_androDi
    }

    try:
        worker = Main(**kwargs)
        worker.main()

    except KeyboardInterrupt as e:
        print('Exit using ctrl^C')
