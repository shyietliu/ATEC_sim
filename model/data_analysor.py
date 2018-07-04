# encoding: utf-8
import csv
import re
import sys
from collections import OrderedDict
reload(sys)
sys.setdefaultencoding('utf8')


DATA_PATH = '/Users/shyietliu/python/project/ATEC/data/atec_nlp_sim_train.csv'

# read data file
with open(DATA_PATH) as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    data = [row for row in reader]

# pattern = re.compile(ur'(\d)\t([\u4e00-\u9fa5]+)\t([\u4e00-\u9fa5])\t(\d)')
punctuation = []
for row in data:
    string_1 = row[1]
    string_2 = row[2]
    res1 = re.sub(u'[\u4e00-\u9fa5]+', '', string_1.decode('utf8'))
    res2 = re.sub(u'[\u4e00-\u9fa5]*', '', string_2.decode('utf8'))
    if res1 is not '':
        res1 = re.sub(r' ', '', res1)
        punctuation.append(res1)
    if res2 is not '':
        res2 = re.sub(r' ', '', res2)
        punctuation.append(res2)
pct = list(OrderedDict.fromkeys(punctuation))
print(pct)
print(len(pct))
pass
