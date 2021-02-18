#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tiago Heinrich

try:
    import argparse
    import sys

    # https://www.journaldev.com/19392/python-xml-to-json-dict
    import xmltodict
    import pprint
    import json

    from os import listdir
    from os.path import isfile, join

except Exception as a:
    print('Import error', a)

class Main(object):

    def __init__(self, folder_name):
        self.folderName = folder_name

    def findFiles(self, path):
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory#3207973
        try:
            data = [str(f) for f in listdir(path) if isfile(join(path, f))]

            return data

        except Exception as a:
            print('main.readFiles Error:', a)

    def main(self):
        try:
            liTotal = []

            dataList = self.findFiles(self.folderName)

            # Atividade 1 lista d permissões

            print('===================\n')
            print('Permissões por APK\n')
            print('===================')

            for eachFile in dataList:

                with open(self.folderName+eachFile) as fd:
                    doc = xmltodict.parse(fd.read(), process_namespaces=True)
                # print(' \__ App:', (doc['manifest']['application'])['@http://schemas.android.com/apk/res/android:name'])
                li = []
                for each in doc['manifest']['uses-permission']:
                    # print(each['@http://schemas.android.com/apk/res/android:name'])
                    # print(((each['@http://schemas.android.com/apk/res/android:name'].split(".")[-1:]))[0])
                    li.append(((each['@http://schemas.android.com/apk/res/android:name'].split(".")[-1:]))[0])

                # print(' \__ Permission:', li)
                print((doc['manifest']['application'])['@http://schemas.android.com/apk/res/android:name'], ':', li)
                liTotal.append(set(li))

            # Atividade 2
            print('===================\n')
            print('Permissões únicas por APK\n')
            print('===================')

            # difference
            # dinter = liTotal[0].symmetric_difference(liTotal[1])
            # for each in liTotal[1:]:
            #     dinter = dinter.symmetric_difference(each)

            for keyB, eachB in enumerate(liTotal):
                tli = []
                for key, each in enumerate(liTotal):
                    if(keyB != key):
                        for add in eachB.difference(each):
                            tli.append(add)
                print((doc['manifest']['application'])['@http://schemas.android.com/apk/res/android:name'], ':', list(set(tli)))

            # print(' \__ Diferença para todos:', list(dinter))

            print('===================\n')
            print('Permissões comuns das APKs\n')
            print('===================')

            # lista de permissões comuns a todas as APKs analisadas (intersection)
            inter = liTotal[0].intersection(liTotal[1])
            for each in liTotal[1:]:
                inter = inter.intersection(each)

            print(list(inter))
            # print(' \__ Comum para todos:', list(inter))

        except Exception as a:
            print('Main.main', a)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='apk tool.')

    parser.add_argument('--version','-v','-vvv','-version', action='version', version=str('Base 0.1'))

    parser.add_argument('--folder-name', type=str, default=' ', required=True, help='This option define the ML method.')

    #get args
    args = parser.parse_args()
    kwargs = {
        'folder_name': args.folder_name
    }

    args = parser.parse_args()

    try:
        worker = Main(**kwargs)
        worker.main()

    except KeyboardInterrupt as e:
        print('Exit using ctrl^C')
