#importing modules
from random import sample
import pyreadr
import pandas as pd

#reading in csv file
#samp_perc = 0.02
chunksize = 100000

tfr = pd.read_csv('full_df_2.csv', chunksize=chunksize, iterator=True, encoding='latin-1')

df = pd.concat(tfr, ignore_index=True)
#df = df.sample(n=round(len(df)*samp_perc), random_state=15)
#print(samp_perc, "of file is being used")
print("Read in data")

#saving as RDS file
pyreadr.write_rds('full_data.RDS', df, compress="gzip")
print('RDS dowloaded!')
