#dlshogi_2019>python utils/convert2222.py kifu kifuv22

import sys
import os
import re
import argparse
import chardet
from chardet.universaldetector import UniversalDetector

parser = argparse.ArgumentParser()
parser.add_argument('read_dir', type=str, help='read csa files from this directory')
parser.add_argument('out_dir', type=str, help='csa v2 output directory')
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

# 指定したディレクトリ配下の棋譜を読み込み、V2形式に変換する。 
for filepath in find_all_files(args.read_dir):
    outfilepath = os.path.join(args.out_dir, os.path.basename(filepath))
    fout = open(outfilepath, 'w', encoding='utf-8')

    try:
        # ファイルのエンコディングを調べてその形式で開く。
        encoding = check_encoding(filepath)
        fin = open(filepath, 'r', encoding=encoding)

        for line in fin.readlines():

          # +と-から始まる行は差し手と処理時間、%は投了などの結果と処理時間
          # V2.2はカンマ区切りだが、V2は改行なのでカンマを改行に変換して出力
          if line[0] in ['-', '+', '%']:
            for _line in line.split(','):
              fout.writelines(_line.strip() + '\n')

          # Vから始まる行はバージョン。V2に書き換えて出力する。
          elif line[0] == 'V' or line[1] == 'V':
            fout.writelines('V2\n')

          # P+やP-など、Pから始まる2文字の行はV2.2形式らしいので無視する。
          elif len(line.strip()) == 2 and line[0] == 'P':
            continue

          # 上記以外はそのまま出力する。
          else:
            fout.writelines(line )
        fout.close()

    except Exception as e: 
        print('[' + filepath + '] ' + str(e) )

