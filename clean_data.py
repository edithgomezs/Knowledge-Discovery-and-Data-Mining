#importing modules
import pandas as pd

#reading in csv file
samp_perc = 0.2
chunksize = 100000
tfr = pd.read_csv('airline.csv', usecols=['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Flight_Number_Reporting_Airline',
                                          'OriginCityName', 'OriginState', 'DestCityName', 'DestState', 'CRSDepTime', 'DepDelay',
                                          'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn', 'CRSArrTime', 'ArrDelay', 'Diverted',
                                          'CRSElapsedTime', 'ActualElapsedTime', 'AirTime', 'Distance', 'Cancelled'], chunksize=chunksize, iterator=True, encoding='latin-1')
df = pd.concat(tfr, ignore_index=True)
df = df.sample(n=round(len(df)*samp_perc), random_state=15)
print(samp_perc, "of file is being used")
print(df.head())

#filtering out cancelled flights
df = df[df['Cancelled'] == 0]
print(df.head())

#dropping Cancelled column
df.drop('Cancelled', axis=1)

#converting Flight_Number_Reporting_Airline to str type
df['Flight_Number_Reporting_Airline'] = df['Flight_Number_Reporting_Airline'].astype(str)

#renaming FlightDate as Date
df.rename(columns={'FlightDate':'Date'}, inplace = True)

#checking size of data
print(df.shape)

#checking the type of each column
print(df.dtypes)

#checking for na values
print(df.isnull().sum())

#dropping na values
df.dropna()
print(df.head())

#checking for na values
print(df.isnull().sum())

#checking size of data
print(df.shape)

#converting military time into minutes
dept = df['CRSDepTime'].tolist()
arrv = df['CRSArrTime'].tolist()
mins_dept = [(((i//100)*60) + (i%100)) for i in dept]
mins_arrv = [(((i//100)*60) + (i%100)) for i in arrv]

df['CRSDepTime'] = mins_dept
df['CRSArrTime'] = mins_arrv

print(df['CRSDepTime'].head())
print(df['CRSArrTime'].head())

#saving as csv file
df.to_csv('0.2airline.csv')
print('csv downloaded!')