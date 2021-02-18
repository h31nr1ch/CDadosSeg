#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Tiago Heinrich

try:
    import argparse
    import sys
    import pefile

except Exception as a:
    print('Import error', a)

class Main(object):

    def __init__(self, file_one, file_two):
        self.file1 = file_one
        self.file2 = file_two

    def anaPE(self, file):
        try:
            pe =  pefile.PE(file)
            header = []
            result = []
            print(file, [f.Name.decode('utf-8') for f in pe.sections] )
            for section in pe.sections:
                header.append(section.Name.decode('utf-8') )

                if ('0x1' in hex(section.Characteristics)):
                    result.append(0)
                elif('0x2' in hex(section.Characteristics)):
                    result.append(1)
                elif('0x3' in hex(section.Characteristics)):
                    result.append(1)
                elif('0x4' in hex(section.Characteristics)):
                    result.append(0)
                elif('0x5' in hex(section.Characteristics)):
                    result.append(0)
                elif('0x6' in hex(section.Characteristics)):
                    result.append(1)
                elif('0x7' in hex(section.Characteristics)):
                    result.append(1)
                elif('0x8' in hex(section.Characteristics)):
                    result.append(0)
                elif('0x9' in hex(section.Characteristics)):
                    result.append(0)
                elif('0xa' in hex(section.Characteristics)): # 10
                    result.append(1)
                elif('0xb' in hex(section.Characteristics)): # 11
                    result.append(1)
                elif('0xc' in hex(section.Characteristics)): # 12
                    result.append(0)
                elif('0xd' in hex(section.Characteristics)): # 13
                    result.append(0)
                elif('0xe' in hex(section.Characteristics)): # 14
                    result.append(1)
                elif('0xf' in hex(section.Characteristics)): # 15
                    result.append(1)
                else:
                    print('sério mesmo?')

            return result, header

        except Exception as a:
            print('Main.anaPE', a)

    def findType(self, index, l1, l2):
        for key, each in enumerate(l1):
            if(index == each):
                if(l2[key] == 1):
                    return 'executável'
                else:
                    return '[não] executável'

    def main(self):
        try:
            fullHeader = []

            results0, header0 = self.anaPE(self.file1)
            a= zip(results0, header0)
            results1, header1 = self.anaPE(self.file2)
            b= zip(results1, header1)

            for each in header0:
                fullHeader.append(each)

            for each in header1:
                fullHeader.append(each)

            print(' ')
            for each in list(set(fullHeader)):
                if(each in header1 and each in header0):
                    if(str(self.findType(each, header1, results1)) == 'executável' and str(self.findType(each, header0, results0)) == 'executável'):
                        print('Ambos arquivos contém a seção', each, 'executável')
                    elif(str(self.findType(each, header1, results1)) == 'executável'):
                        print('Ambos arquivos contém a seção', each, 'mas somente o', self.file2, 'é executável')
                    elif(str(self.findType(each, header0, results0)) == 'executável'):
                        print('Ambos arquivos contém a seção', each, 'mas somente o ',self.file1,' é executável')
                    else:
                        print('Ambos arquivos contém a seção', each, '[não] executável')
                elif(each in header1):
                    print('Somente ', self.file2 ,' contém a seção', each, str(self.findType(each, header1, results1)))
                elif(each in header0):
                    print('Somente ',self.file1,' contém a seção', each, str(self.findType(each, header0, results0)))
                else:
                    print(' :? ')

        except Exception as a:
            print('Main.main', a)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PE tool.')

    parser.add_argument('--version','-v','-vvv','-version', action='version', version=str('Base 0.1'))

    parser.add_argument('--file-one', type=str, default=' ', required=True, help='.')

    parser.add_argument('--file-two', type=str, default=' ', required=True, help='.')

    #get args
    args = parser.parse_args()
    kwargs = {
        'file_one': args.file_one,
        'file_two': args.file_two
    }

    args = parser.parse_args()

    try:
        worker = Main(**kwargs)
        worker.main()

    except KeyboardInterrupt as e:
        print('Exit using ctrl^C')
