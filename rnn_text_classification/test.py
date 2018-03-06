# coding:utf-8
import sys
sys.path.append("..")
from rnn_text_classification.rnnclassify_transfer import Classify_CN
import os
import pandas as pd
from pprint import pprint
from sklearn.metrics import accuracy_score
from sys import argv

# script, module_path, test_data_path, embed_way = argv
module_path = '../work_space/suning/module/comb_hlt'
test_data_path = '../work_space/suning/dataset/suning_biu_dialogue_data'
embed_way = 'hlt'

classify = Classify_CN(module_path, embed_way)

input = []
output_actu = []
for parent,dirnames,filenames in os.walk(test_data_path):
    # print(filenames)
    for filename in filenames:
        if filename[-1] == '~':
            continue
        else:
            for line in open(os.path.join(parent, filename)):
                line_clean = line.strip()
                input.append(line_clean)
                output_actu.append(filename)

output_pred = []
# time_cost = []
count = 0
for sentence in input:
    result = classify.getCategory(sentence)
    output_pred.append(result['value'])
    # time_cost.append(time)
    # time_aver = sum(time_cost)/len(time_cost)

    if result['value'] != output_actu[count]:
        print(sentence)
        print(result)
        print('Pred label: ' + result['value'])
        print('Actual label: ' + output_actu[count] + '\n')
    count += 1

y_actu = pd.Series(output_actu, name='Actual')
y_pred = pd.Series(output_pred, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, margins=True)
accuracy = accuracy_score(y_actu, y_pred)

print('plot confusion matrix:\n')
pprint(df_confusion)
print('Accuracy: {}'.format(accuracy))
# print('Average time:{}'.format(time_aver))
