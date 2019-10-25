import argparse
import os
import re
import statistics
from chardet.universaldetector import UniversalDetector

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
args = parser.parse_args()

# 文字コード判定
def check_encoding(file_path):
    detector = UniversalDetector()
    with open(file_path, mode='rb') as f:
        for binary in f:
            detector.feed(binary)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']

# ディレクトリ配下の全てのファイルを取得する(サブディレクトリも)。
def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)
            
kifu_count = 0
file = []

#print(args.dir)
for filepath in find_all_files(args.dir):
    #print(filepath)

    # ファイルのエンコディングを調べてその形式で開く。
    encoding = check_encoding(filepath)
    #print(filepath)
    fin = open(filepath, 'r', encoding=encoding)
    toryo = False
        
    for line in fin.readlines():
        #print(line)
        line = line.strip()
        #print(line)

        if line == '%TORYO':
            toryo = True
            
    if not toryo: 
        print(filepath)
        file.append(filepath)
        #os.remove(filepath)
        kifu_count += 1
        
    
print('kifu count :', kifu_count)
print(file)
for file_ in file:
    os.remove(file_)