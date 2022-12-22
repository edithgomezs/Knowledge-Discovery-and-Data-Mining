import pandas as pd
import pyreadr

chunksize = 10000

##### Read in our data #####
print("Starting the joining process")
# weather data
result = pyreadr.read_r('weather_df_clean.RDS')
weather_df = result[None]
print("read in weather data")

# airline data
#airline_df = pd.read_csv('0.2airline.csv', encoding='latin-1', on_bad_lines='skip')
#print("read in airline data")
##############################################################################################

##### Clean data #####
# Airline Drop cancelled rows
# airline_df = airline_df[airline_df['Cancelled'] == 0]
# airline_df = airline_df.drop('Cancelled', axis=1)

# Weather drop GUST and SLP and name
weather_df = weather_df.drop('GUST', axis=1)
weather_df = weather_df.drop('SLP', axis=1)
weather_df = weather_df.drop('NAME', axis=1)


# Drop everything after comman including comma airline city names
# airline_df["OriginCityName"] = airline_df["OriginCityName"].replace(',.*$','', regex=True)
# airline_df["DestCityName"] = airline_df["DestCityName"].replace(',.*$','', regex=True)

# Change date cols to datatime objects
weather_df["DATE"] = pd.to_datetime(weather_df["DATE"], errors='coerce', infer_datetime_format=True)
# airline_df["Date"] = pd.to_datetime(airline_df["Date"], errors='coerce', infer_datetime_format=True)


##############################################################################################
##### Join our two datasets into one full one #####

i = 0
def preprocess(airline_df):
    global i
    # remove cancelled flights and airprot name
    airline_df = airline_df[airline_df['Cancelled'] == 0]
    airline_df = airline_df.drop('Cancelled', axis=1)

    # drop everything after commas
    airline_df["OriginCityName"] = airline_df["OriginCityName"].replace(',.*$','', regex=True)
    airline_df["DestCityName"] = airline_df["DestCityName"].replace(',.*$','', regex=True)
    
    # convert to datetime
    airline_df["Date"] = pd.to_datetime(airline_df["Date"], errors='coerce', infer_datetime_format=True)
    
    # join for origin
    full_df_origin = pd.merge(weather_df, airline_df, left_on=["DATE", "CITY", "STATE"], right_on=["Date", "OriginCityName", "OriginState"], how='inner')
    # rename weather columns for origin
    full_df_origin.rename(columns={'ELEVATION': 'ELEVATION_origin', 'TEMP': 'TEMP_origin', 'DEWP': 'DEWP_origin', 'STP': 'STP_origin', 'VISIB': 'VISIB_origin', 'WDSP': 'WDSP_origin', 
    'MXSPD': 'MXSPD_origin', 'MAX': 'MAX_origin', 'MIN': 'MIN_origin', 'PRCP': 'PRCP_origin', 'SNDP': 'SNDP_origin', 'LATITUDE': 'LATITUDE_origin', 'LONGITUDE': 'LONGITUDE_origin',
    'Clear': 'Clear_origin', 'Fog': 'Fog_origin', 'Rain': 'Rain_origin', 'Snow': 'Snow_origin', 'Hail': 'Hail_origin', 'Thunder': 'Thunder_origin', 'Tornado': 'Tornada_origin'}, inplace=True)

    # join dest data
    full_df = pd.merge(weather_df, full_df_origin, left_on=["DATE", "CITY", "STATE"], right_on=["Date", "DestCityName", "DestState"], how='inner')
    # rename weather columns for dest
    full_df.rename(columns={'ELEVATION': 'ELEVATION_dest', 'TEMP': 'TEMP_dest', 'DEWP': 'DEWP_dest', 'STP': 'STP_dest', 'VISIB': 'VISIB_dest', 'WDSP': 'WDSP_dest', 
    'MXSPD': 'MXSPD_dest', 'MAX': 'MAX_dest', 'MIN': 'MIN_dest', 'PRCP': 'PRCP_dest', 'SNDP': 'SNDP_dest', 'LATITUDE': 'LATITUDE_dest', 'LONGITUDE': 'LONGITUDE_dest',
    'Clear': 'Clear_dest', 'Fog': 'Fog_dest', 'Rain': 'Rain_dest', 'Snow': 'Snow_dest', 'Hail': 'Hail_dest', 'Thunder': 'Thunder_dest', 'Tornado': 'Tornada_dest'}, inplace=True)

    # on first run, add header. Else dont add header
    if i == 0:
        full_df.to_csv("full_df_2.csv",mode="a",header=True,index=False)
    else:
        full_df.to_csv("full_df_2.csv",mode="a",header=False,index=False)
    i = i + 1

reader = pd.read_csv('0.2airline.csv', encoding='latin-1', on_bad_lines='skip', chunksize = chunksize) # chunksize depends with you colsize

#temp_airline = pd.read_csv('0.2airline.csv', encoding='latin-1', on_bad_lines='skip', nrows=1) # chunksize depends with you colsize

#temp_header = pd.merge(weather_df, temp_airline, left_on=["CITY"], right_on=["DestState"], how='inner')  # incorrect join to just get headers
#temp_header.to_csv("full_df_2.csv",mode="a",header=True,index=False)  # add headers to CSV


[preprocess(r) for r in reader]  # we need to read in by chuck size since data takes up too much memory
