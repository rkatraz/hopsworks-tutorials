import math
import requests
import joblib
from datetime import datetime, timedelta
import zipfile
from io import BytesIO
import os
import json
from calendar import monthrange
import pandas as pd
import time

# Mute warnings
import warnings
warnings.filterwarnings("ignore")

###############################################################################
# Basic functions

def convert_date_to_unix(x):
    dt_obj = datetime.strptime(str(x), '%Y-%m-%d')
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)


def convert_unix_to_date(x):
    x //= 1000
    x = datetime.fromtimestamp(x)
    return datetime.strftime(x, "%Y-%m-%d")


def get_next_date(date):
    next_date = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)
    return datetime.strftime(next_date, "%Y-%m-%d")


def get_last_date_in_fg(fg):
    for col in fg.statistics.content["columns"]:
        if col["column"] == "timestamp":
            res = col["maximum"]
            return convert_unix_to_date(res)



###############################################################################
# Data basic processing

def select_stations_info(df):
    df_res = df[["start_station_id", "start_station_name",
                 "start_lat", "start_lng"]].rename(columns={"start_station_id": "station_id",
                                                            "start_station_name": "station_name",
                                                            "start_lat": "lat",
                                                            "start_lng": "long"
                                                                        })
    df_res = df_res.drop_duplicates()
    return df_res


def process_df(original_df, month, year):
    df_res = original_df[["started_at", "start_station_id"]]
    df_res.started_at = pd.to_datetime(df_res.started_at)
    df_res.started_at = df_res.started_at.dt.floor('d')
    df_res = df_res.groupby(["started_at",
                             "start_station_id"]).value_counts().reset_index()
    df_res = df_res.rename(columns={"started_at": "date",
                                    "start_station_id": "station_id",
                                    0: "users_count"})
    # lets select only popular station - the station will be considered a
    # popular one, if it was used every day for each month
    days_in_month = monthrange(int(year), int(month))[1]
    popular_stations = df_res.station_id.value_counts()[df_res.station_id.value_counts() == days_in_month].index
    df_res = df_res[df_res.station_id.isin(popular_stations)]
    return df_res.sort_values(by=["date", "station_id"])


def update_month_data(main_df, month, year):
    if month < 10:
        month = f"0{month}"
    print(f"_____ Processing {month}/{year}... _____")

    if f"{year}{month}" in ["202206", "202207"]:
        citibike = "citbike"
    else:
        citibike = "citibike"
    url = f'https://s3.amazonaws.com/tripdata/{year}{month}-{citibike}-tripdata.csv.zip'
    print(url)
    filename = "data/" + url.split('/')[-1].split(".")[0] + ".csv"
    fn_list = filename.split(".")
    processed_filename = fn_list[-2] + "_processed." + fn_list[-1]

    if os.path.isfile(processed_filename):
        print("Retrieving DataFrame from the existing csv file...💿")
        return pd.concat([pd.read_csv(processed_filename), main_df])

    print('Downloading Started...⏳')

    req = requests.get(url)

    print('Downloading Completed!👌')

    file= zipfile.ZipFile(BytesIO(req.content))

    file.extractall("data/")

    print("Retrieving DataFrame from the csv file...💿")
    print("-" * 32)

    original_df = pd.read_csv(filename)

    if not os.path.isfile("data/stations_info.csv"):
        stations_info_df = select_stations_info(original_df)
    else:
        present_stations_df = pd.read_csv("data/stations_info.csv")
        stations_info_batch = select_stations_info(original_df)
        stations_info_df = pd.concat([
            present_stations_df,
            stations_info_batch
            ]).reset_index(drop=True)
    stations_info_df.to_csv("data/stations_info.csv", index=False)

    processed_df = process_df(original_df, month, year)

    # delete original big unprocessed file
    os.remove(filename)

    # save processed file in csv
    processed_df.to_csv(processed_filename, index=False)

    return pd.concat([processed_df, main_df])


def get_citibike_data(start_date="04/2021", end_date="10/2022") -> pd.DataFrame:

    start_month, start_year = start_date.split("/")[0], start_date.split("/")[1]
    end_month, end_year = end_date.split("/")[0], end_date.split("/")[1]

    df_res = pd.DataFrame(columns=["date", "station_id", "users_count"])

    if start_year == end_year:
        for month in range(int(start_month), int(end_month) + 1):
            df_res =  update_month_data(df_res, month, start_year)
    else:
        for month in range(int(start_month), 12 + 1):
            df_res =  update_month_data(df_res, month, start_year)
        for month in range(1, int(end_month) + 1):
            df_res =  update_month_data(df_res, month, end_year)

    df_res["users_count"] = df_res["users_count"].astype(int)

    print("\n✅ Done ✅")

    return df_res.reset_index(drop=True)


###############################################################################
# Weather parsing

def get_weather_data_from_open_meteo(city_name: str,
                                     start_date: str,
                                     end_date: str,
                                     coordinates: list,
                                     forecast: bool = False):
    """
    Takes city name, coordinates and returns pandas DataFrame with weather data.
    
    Examples of arguments:
        coordinates=(47.755, -122.2806), start_date="2023-01-01"
    """
    start_of_cell = time.time()
    
    latitude, longitude = coordinates
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'daily': ["temperature_2m_max", "temperature_2m_min",
                  "precipitation_sum", "rain_sum", "snowfall_sum",
                  "precipitation_hours", "windspeed_10m_max",
                  "windgusts_10m_max", "winddirection_10m_dominant"],
        'start_date': start_date,
        'end_date': end_date,
        'timezone': "Europe/London"
    }
    
    if forecast:
        # historical forecast endpoint
        base_url = 'https://api.open-meteo.com/v1/forecast' 
    else:
        # historical observations endpoint
        base_url = 'https://archive-api.open-meteo.com/v1/archive'  
        
    try:
        response = requests.get(base_url, params=params)
    except ConnectionError:
        response = requests.get(base_url, params=params)
    
    response_json = response.json()    
    res_df = pd.DataFrame(response_json["daily"])
    
    # rename columns
    res_df = res_df.rename(columns={
        "time": "date",
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "windspeed_10m_max": "wind_speed_max",
        "winddirection_10m_dominant": "wind_direction_dominant",
        "windgusts_10m_max": "wind_gusts_max"
    })
    
    # change columns order
    res_df = res_df[
        ['date', 'temperature_max', 'temperature_min',
         'precipitation_sum', 'rain_sum', 'snowfall_sum',
         'precipitation_hours', 'wind_speed_max',
         'wind_gusts_max', 'wind_direction_dominant']
    ]
    
    # convert dates in 'date' column
    res_df["date"] = pd.to_datetime(res_df["date"])
    end_of_cell = time.time()
    print(f"Parsed weather for {city_name} since {start_date} till {end_date}.")
    print(f"Took {round(end_of_cell - start_of_cell, 2)} sec.\n")
        
    return res_df

################################################################################
# Data engineering

def moving_average(df, window=7):
    df[f'mean_{window}_days'] = df.groupby('station_id')['users_count'] \
                                    .rolling(window=window).mean().reset_index(0,drop=True).shift(1)
    return df

# def moving_average(df, window=7):
#     df[f'mean_{window}_days'] = df["users_count"].rolling(window=window).mean()
#     return df


def moving_std(df, window):
    df[f'std_{window}_days'] = df.groupby('station_id')['users_count'] \
                                    .rolling(window=window).std().reset_index(0,drop=True).shift(1)
    return df


def exponential_moving_average(df, window):
    df[f'exp_mean_{window}_days'] = df.groupby('station_id')['users_count'].ewm(span=window) \
                                        .mean().reset_index(0,drop=True).shift(1)
    return df


def exponential_moving_std(df, window):
    df[f'exp_std_{window}_days'] = df.groupby('station_id')['users_count'].ewm(span=window) \
                                        .std().reset_index(0,drop=True).shift(1)
    return df


def engineer_citibike_features(df):
    df_res = df.copy()
    # there are duplicated rows (several records for the same day and station). get rid of it.
    df_res = df_res.groupby(['date', 'station_id'], as_index=False)['users_count'].sum()

    df_res['prev_users_count'] = df_res.groupby('station_id')['users_count'].shift(+1)
    df_res = df_res.dropna()
    df_res = moving_average(df_res, 7)
    df_res = moving_average(df_res, 14)


    for i in [7, 14]:
        for func in [moving_std, exponential_moving_average,
                     exponential_moving_std
                     ]:
            df_res = func(df_res, i)
    df_res = df_res.reset_index(drop=True)
    return df_res.sort_values(by=["date", "station_id"]).dropna()

###############################################################################
# Streamlit
def get_model(project, model_name, file_name):
    # load our Model
    list_of_files = [os.path.join(dirpath,filename) for dirpath, _, filenames in os.walk('.') for filename in filenames if filename == file_name]

    if list_of_files:
        model_path = list_of_files[0]
        model = joblib.load(model_path)
    else:
        if not os.path.exists(file_name):
            mr = project.get_model_registry()
            EVALUATION_METRIC="r2_score"
            SORT_METRICS_BY="max"
            # get best model based on custom metrics
            model = mr.get_best_model(model_name,
                                      EVALUATION_METRIC,
                                      SORT_METRICS_BY)
            model_dir = model.download()
            model = joblib.load(model_dir + f"/{file_name}")

    return model
