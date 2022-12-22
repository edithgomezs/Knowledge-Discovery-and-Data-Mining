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
from sklearn.model_selection import RandomizedSearchCV

#### Read in our data ####
result = pyreadr.read_r('full_data_clean.RDS')
df = result[None]

#df = pd.read_csv('full_data_clean_n10000.csv')

df_temp = pd.DataFrame(columns=['sample_perc', 'time', 'f1', 'accuracy'])  # fake empty dataset to add header to csv
df_temp.to_csv("results.csv", mode="w", header=True, index=False)


#### Data Cleaning ####
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']  # we only want numeric data
df = df.select_dtypes(include=numerics)


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
sample_percs = [.50]  # what perc do we want to undersample ontime class by

for sample_perc in sample_percs:
    train, test = train_test_split(df, test_size=0.2)  # Split data 20% test, 80% train

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

    #### Now we want to implament gridsearch ####

    # set our paramiter ranges 
    n_estimators = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # Number of trees in random forest
    max_features = ['auto', 'sqrt', 'log2']  # The number of features to consider when looking for the best split
    max_depth = [20, 40, 60, 80, 90, 100, None]  # The maximum depth of the tree
    min_samples_split = [2, 3, 4, 5, 6]  # The minimum number of samples required to split an internal node:
    criterion = ['gini', 'entropy']  # The function to measure the quality of a split

    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'criterion': criterion}

    #start = timeit.default_timer()  # start timer
    n_iter = 3  # n_iter was 100
    cv = 2

    clf = RandomForestClassifier()  # our RF model. Off the shelf    
    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = n_iter, cv = cv, verbose=2, n_jobs = -1) 
    
    
    rf_random.fit(train, y_train)  # training
    y_pred = rf_random.predict(test)  # predictions
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    acc = metrics.accuracy_score(y_test, y_pred)

    print('#########################################################################################################################')
    print(f'N_iter: {n_iter}')
    print(f'CV: {cv}')
    print(f'Acc: {acc}')
    print(f'F1: {f1}')
    print(f'Best params: {rf_random.best_params_}')

    

    # y_pred = clf.predict(test)  # predictions
    
    # f1 = metrics.f1_score(y_test, y_pred, average='macro')
    # acc = metrics.accuracy_score(y_test, y_pred)

    # stop = timeit.default_timer()  # end timer

    # # write results to csv
    # with open(f"results.csv", "a+", newline='') as csvfile:
    #     csvwriter= csv.writer(csvfile)
    #     csvwriter.writerows([[sample_perc, stop - start, f1, acc]])
    
    # # create feature importance chart
    # sorted = rf_random.feature_importances_.argsort()
    # plt.figure(figsize=(17, 10))
    # plt.barh(train.columns[sorted], clf.feature_importances_[sorted])
    # plt.xlabel("Permutation Importance")
    # plt.savefig(f'Permutation_Importance_{cv}_{acc}.png')



    # return sample_perc || this tells us the percent we undersampled the on time class by








