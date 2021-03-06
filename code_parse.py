# from pprint import pprint
import os
import ast
from ast2json import str2json
import pickle

def code_parse(filename):
    with open(filename) as f:
        file = f.readlines()
        
    # remove the first two and the last columns
    print('-- strip')
    textstrip = [line.split('\t')[2:-1] for line in file]

    # remove lines that are not made of two parts
    print('-- remove irregular lines')
    outliners = []
    for i, line in enumerate(textstrip):
        try:
            assert(len(line) == 2)
        except AssertionError:
            outliners.append(i)
    textinlier = [line for i, line in enumerate(textstrip) if i not in outliners]

    # unescape \n
    print('-- unescape \\n')
    textreplace = [[comment, code.replace('\\n', '\n')] for comment, code in textinlier]

    print('-- parse and remove failures')
    # remove the examples that can not be parsed to ast
    text = []
    failed = 0
    for i, [comment, code] in enumerate(textreplace):
        print('\tTotal: %i Failed: %i' % (i, failed), end='\r')
        try:
            code_js = str2json(code)
        except SyntaxError:
            # convert to python3
            with open('snippet.py', 'w+') as f:
                f.write(code)
            os.system('2to3 -w snippet.py 1>/dev/null 2>&1')
            code = open('snippet.py').read()
            try:
                code_js = str2json(code)
            except SyntaxError:
                failed += 1
                # print('parse failed!')
                continue
        text.append((comment, code))
    os.system('rm snippet.py')
    print()

    print('-- save')
    phase = os.path.splitext(os.path.basename(filename))[0]
    with open('data/{}.pkl'.format(phase), 'wb') as f:
        pickle.dump(text, f)

if __name__ == '__main__':
    print('\nParse test set..')
    code_parse('raw_data/test.txt')

    print('\nParse train set..')
    code_parse('raw_data/train.txt')

    print('\nParse valid set..')
    code_parse('raw_data/valid.txt')
