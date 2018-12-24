import pandas as pd
import numpy as np

users = pd.read_csv("data/ads_dataset.tsv",sep = '\t')
users.head()

def getDfSummary(input_data):
    output_data = input_data.describe()
    output_data = output_data.T
    output_data['spread'] = output_data['max']-output_data['min']
    return output_data

output = getDfSummary(users)
output.loc[output['spread']==1.00000]


#print(output)

def squared_less_100(input_number):
    # Code here!
    output_number = np.square(input_number)-100
    return output_number

sl1_results =list()
absolute_results =list()
log_results =list()

oringinal_set = np.array(range(1,21))
print(oringinal_set)

for i in oringinal_set:
    sl1_results.append(squared_less_100(i))

for i in oringinal_set:
    absolute_results.append(np.abs(i))

for i in oringinal_set:
    log_results.append(np.log(i))

print(sl1_results)
print(absolute_results)
print(log_results)


