
import os

root = '/Users/yvan/data/voc2007/labels'
s = sorted(os.listdir(root))
for i in s:
    with open('train.txt','a+') as f:
        f.write(root + '/' + i + '\n')
        f.close()