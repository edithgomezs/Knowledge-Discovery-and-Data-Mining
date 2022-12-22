from email import header
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import pyreadr
import timeit
import numpy as np
import csv

#### Read in our data ####
result = pyreadr.read_r('full_data_clean.RDS')
df = result[None]

#df = pd.read_csv('full_data_clean_n10000.csv')

df_temp = pd.DataFrame(columns=['sample_perc', 'time', 'f1', 'accuracy'])  # fake empty dataset to add header to csv
df_temp.to_csv("results.csv", mode="w", header=True, index=False)


#### Data Cleaning ####
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  # we only want numeric data
df = df.select_dtypes(include=numerics)

df.drop('CRSElapsedTime', axis=1, inplace=True) # drop taxiout
df.drop('AirTime', axis=1, inplace=True) # drop taxiout
df.drop('TaxiOut', axis=1, inplace=True) # drop taxiout
df.drop('WheelsOff', axis=1, inplace=True) # drop WheelsOff
df.drop('WheelsOn', axis=1, inplace=True) # drop WheelsOn
df.drop('TaxiIn', axis=1, inplace=True) # drop taxiout
df.drop('ActualElapsedTime', axis=1, inplace=True) # drop ActualElapsedTime

df['delay'] = np.where(df['DepDelay'] >= 15.0, '1', '0')  # if dept time delyed more then 15 min, mark it as delayed 
df.drop('DepDelay', axis=1, inplace=True) # drop DepDelay

df['delay'] = np.where(df['ArrDelay'] >= 15.0, '1', '0')  # if arrive time delyed more then 15 min, mark it as delayed 
df.drop('ArrDelay', axis=1, inplace=True) # drop ArrDelay

print(df['delay'].value_counts())  # examine class imbalence


#### Now time for the fun part, modeling!! ####
sample_percs = [1, .75, .50, .25]  # what perc do we want to undersample ontime class by

for sample_perc in sample_percs:
    train, test = train_test_split(df, test_size=0.2, random_state=42)  # Split data 20% test, 80% train

    # we will only under sample not delyed since we will have a faster run time
    delayed = train.loc[(train['delay'] == '1')]  # all delayed instances
    on_time = train.loc[(train['delay'] == '0')]  # all on time instances
    
    temp_on_time = on_time.sample(n=round(len(on_time) * sample_perc))  # randomly sample
    frames = [delayed, temp_on_time]
    train = pd.concat(frames)

    y_train = train['delay']
    train.drop('delay', axis=1, inplace=True) # drop target var
    y_test = test['delay']
    test.drop('delay', axis=1, inplace=True)  # drop target var

    start = timeit.default_timer()  # start timer
    clf = RandomForestClassifier()  # our RF model. Off the shelf
    clf.fit(train, y_train)  # training
    y_pred = clf.predict(test)  # predictions
    
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    acc = metrics.accuracy_score(y_test, y_pred)

    stop = timeit.default_timer()  # end timer

    # write results to csv
    with open(f"results.csv", "a+", newline='') as csvfile:
        csvwriter= csv.writer(csvfile)
        csvwriter.writerows([[sample_perc, stop - start, f1, acc]])
    
    # create feature importance chart
    sorted = clf.feature_importances_.argsort()
    plt.figure(figsize=(17, 10))
    plt.barh(train.columns[sorted], clf.feature_importances_[sorted])
    plt.xlabel("Permutation Importance")
    plt.savefig(f'Permutation_Importance_{sample_perc * 100}.png')



    # return sample_perc || this tells us the percent we undersampled the on time class by








