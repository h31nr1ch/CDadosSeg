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

    def __init__(self, file_one):
        self.file1 = file_one

    def anaPE(self, file):
        try:
            pe =  pefile.PE(file)
            header = []
            result = []

            print(file, [f.Name.decode('utf-8') for f in pe.sections] )
            for section in pe.sections:
                print(section.Name.decode('utf-8'))
                header.append(section.Name.decode('utf-8') )

                if ('0x1' in hex(section.Characteristics)):
                    print('  [não] é executável: The section can be shared in memory.')
                    result.append(0)
                elif('0x2' in hex(section.Characteristics)):
                    print('  é executável: The section can be executed as code.')
                    result.append(1)
                elif('0x3' in hex(section.Characteristics)):
                    print('  é executável: shared in memory & executed as code.')
                    result.append(1)
                elif('0x4' in hex(section.Characteristics)):
                    print('  [não] é executável: The section can be read.')
                    result.append(0)
                elif('0x5' in hex(section.Characteristics)):
                    print('  [não] é executável: read & shared in memory.')
                    result.append(0)
                elif('0x6' in hex(section.Characteristics)):
                    print('  é executável: read & executed as code.')
                    result.append(1)
                elif('0x7' in hex(section.Characteristics)):
                    print('  é executável: shared in memory & read & executed as code.')
                    result.append(1)
                elif('0x8' in hex(section.Characteristics)):
                    print('  [não] é executável: The section can be written to.')
                    result.append(0)
                elif('0x9' in hex(section.Characteristics)):
                    print('  [não] é executável: written & shared memory.')
                    result.append(0)
                elif('0xa' in hex(section.Characteristics)): # 10
                    print('  é executável: writen & executed.')
                    result.append(1)
                elif('0xb' in hex(section.Characteristics)): # 11
                    print('  é executável: writen & executed & shared memory.')
                    result.append(1)
                elif('0xc' in hex(section.Characteristics)): # 12
                    print('  [não] é executável: writen & read.')
                    result.append(0)
                elif('0xd' in hex(section.Characteristics)): # 13
                    print('  [não] é executável: writen & read & shared in memory.')
                    result.append(0)
                elif('0xe' in hex(section.Characteristics)): # 14
                    print('  é executável: writen & read & executed.')
                    result.append(1)
                elif('0xf' in hex(section.Characteristics)): # 15
                    print('  é executável: writen & read & executed & shared in memory.')
                    result.append(1)
                else:
                    print('sério mesmo?')

                print(' ')
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

        except Exception as a:
            print('Main.main', a)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PE tool.')

    parser.add_argument('--version','-v','-vvv','-version', action='version', version=str('Base 0.1'))

    parser.add_argument('--file-one', type=str, default=' ', required=True, help='.')

    #get args
    args = parser.parse_args()
    kwargs = {
        'file_one': args.file_one
    }

    args = parser.parse_args()

    try:
        worker = Main(**kwargs)
        worker.main()

    except KeyboardInterrupt as e:
        print('Exit using ctrl^C')
