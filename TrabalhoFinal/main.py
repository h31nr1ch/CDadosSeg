#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tiago Heinrich & Rodrigo Lemos

try:
    import argparse
    import sys

    import numpy as np
    import pandas as pd

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn import preprocessing

except Exception as a:
    print('Import error', a)

class Main():
    def __init__(self, dataset_file):
        self.dataset = dataset_file

    def main(self):
        # Read data
        df = pd.read_csv(self.dataset, header = 0, delimiter = ",")
        # Drop Apk name
        del df['Apk']
        # Drop and copy label
        y = np.array(df['Label'])
        del df['Label']

        # sparce matrix
        # vectorizer = CountVectorizer()
        # X = vectorizer.fit_transform(df)
        # print(X.toarray())

        # Normalization
        min_max_scaler = preprocessing.MinMaxScaler()
        final = pd.DataFrame(min_max_scaler.fit_transform(df))

        print(final)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CDS Trabalho Final.')

    parser.add_argument('--version','-v','-vvv','-version', action='version', version=str('Base 0.1'))

    parser.add_argument('--dataset-file', type=str, required=True, help='Input dataset file.')

    args = parser.parse_args()
    kwargs = {
        'dataset_file': args.dataset_file
    }

    try:
        worker = Main(**kwargs)
        worker.main()

    except KeyboardInterrupt as e:
        print('Exit using ctrl^C')
