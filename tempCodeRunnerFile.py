import pandas as pd

train0_data = pd.read_json('data/train_0.json')
train1_data = pd.read_json('data/train_1.json')
train2_data = pd.read_json('data/train_2.json')
train3_data = pd.read_json('data/train_3.json')
train4_data = pd.read_json('data/train_4.json')
test_data = pd.read_json('data/test.json')
print(train0_data.head())
print(train1_data.head())
print(train2_data.head())
print(train3_data.head())
print(train4_data.head())