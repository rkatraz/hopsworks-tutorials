import json
import time
import pickle
import joblib

import hopsworks 
import streamlit as st
from geopy import distance

import plotly.express as px
import folium
from streamlit_folium import st_folium

from functions import *
import features.air_quality



def print_fancy_header(text, font_size=22, color="#ff5f27"):
    res = f'<span style="color:{color}; font-size: {font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)  
       

# I want to cache this so streamlit would run much faster after restart (it restarts a lot)
@st.cache_data()
def get_feature_view():
    st.write("Getting the Feature View...")
    feature_view = fs.get_feature_view(
        name = 'air_quality_fv',
        version = 1
    )
    st.write("✅ Success!")

    return feature_view
    

@st.cache_data()
def get_batch_data_from_fs(td_version, date_threshold):
    st.write(f"Retrieving the Batch data since {date_threshold}")
    feature_view.init_batch_scoring(training_dataset_version=td_version)

    batch_data = feature_view.get_batch_data(start_time=date_threshold)
    return batch_data


@st.cache_data()
def download_model(name="air_quality_xgboost_model",
                   version=1):
    mr = project.get_model_registry()
    retrieved_model = mr.get_model(
        name="air_quality_xgboost_model",
        version=1
    )
    saved_model_dir = retrieved_model.download()
    return saved_model_dir


@st.cache_data()
def parse_aq_data(last_dates_dict, today):
    start_of_cell = time.time()
    df_aq_raw = pd.DataFrame()
    
    st.write("Parsing started...")
    for continent in target_cities:
        for city_name, coords in target_cities[continent].items():
            df_ = get_aqi_data_from_open_meteo(city_name=city_name,
                                               coordinates=coords,
                                               start_date=last_dates_dict[city_name],
                                               end_date=str(today))
            df_aq_raw = pd.concat([df_aq_raw, df_]).reset_index(drop=True)
    end_of_cell = time.time()
    print("-" * 64)
    st.write(f"Parsed new PM2.5 data for ALL locations up to {str(today)}.")
    st.write(f"Took {round(end_of_cell - start_of_cell, 2)} sec.\n")
    return df_aq_raw


@st.cache_data()
def parse_weather(last_dates_dict, today):
    df_weather_update = pd.DataFrame()
    start_of_cell = time.time()
    
    st.write("Parsing started...")
    for continent in target_cities:
        for city_name, coords in target_cities[continent].items():
            df_ = get_weather_data_from_open_meteo(city_name=city_name,
                                                   coordinates=coords,
                                                   start_date=last_dates_dict[city_name],
                                                   end_date=str(today),
                                                   forecast=True)
            df_weather_update = pd.concat([df_weather_update, df_]).reset_index(drop=True)

    end_of_cell = time.time()
    print(f"Parsed new weather data for ALL cities up to {str(today)}.")
    print(f"Took {round(end_of_cell - start_of_cell, 2)} sec.\n")
    return df_weather_update


def plot_pm2_5(df):
    # create figure with plotly express
    fig = px.line(df, x='date', y='pm2_5', color='city_name')

    # customize line colors and styles
    fig.update_traces(mode='lines+markers')
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'legend_title': 'City',
        'legend_font': {'size': 12},
        'legend_bgcolor': 'rgba(0, 0, 0, 0)',
        'xaxis': {'title': 'Date'},
        'yaxis': {'title': 'PM2.5'},
        'shapes': [{
            'type': 'line',
            'x0': datetime.datetime.now().strftime('%Y-%m-%d'),
            'y0': 0,
            'x1': datetime.datetime.now().strftime('%Y-%m-%d'),
            'y1': df['pm2_5'].max(),
            'line': {'color': 'black', 'width': 2, 'dash': 'dashdot'}
        }]
    })

    # show plot
    st.plotly_chart(fig, use_container_width=True)


with open('target_cities.json') as json_file:
    target_cities = json.load(json_file)


#########################
st.title('🌫 Air Quality Prediction 🌦')

st.write(3 * "-")
print_fancy_header('\n📡 Connecting to Hopsworks Feature Store...')

st.write("Logging... ")
# (Attention! If the app has stopped at this step,
# please enter your Hopsworks API Key in the commmand prompt.)
project = hopsworks.login()
fs = project.get_feature_store()
st.write("✅ Logged in successfully!")

feature_view = get_feature_view()

# I am going to load data for of last 60 days (for feature engineering)
today = datetime.date.today()
date_threshold = today - datetime.timedelta(days=60)

st.write(3 * "-")
print_fancy_header('\n☁️ Retriving batch data from Feature Store...')
batch_data = get_batch_data_from_fs(td_version=1,
                                    date_threshold=date_threshold)

st.write("Batch data:")
st.write(batch_data.sample(5))
st.write(3 * "-")

last_dates_dict = batch_data[["date", "city_name"]].groupby("city_name").max()
last_dates_dict.date = last_dates_dict.date.astype(str)
# here is a dictionary with city names as keys and last updated date as values
last_dates_dict = last_dates_dict.to_dict()["date"]   

if str(today) == max(last_dates_dict.values()):
    print_fancy_header("\n ⏱ Everything is up-to-date!")
else:
    print_fancy_header('\n🌫 Parsing Air Quality data (PM2.5)')


    df_aq_raw = parse_aq_data(last_dates_dict, today)

    # we need the previous data to calculate aggregation functions
    df_aq_update = pd.concat([
        batch_data[df_aq_raw.columns],
        df_aq_raw
    ]).reset_index(drop=True)
    df_aq_update = df_aq_update.drop_duplicates(subset=['city_name', 'date'])

    st.write(df_aq_update.tail(7))

    print_fancy_header(text='\n🛠 Feature Engineering the PM2.5',
                       font_size=18, color="#FDF4F5")

    ###
    df_aq_update['date'] = pd.to_datetime(df_aq_update['date'])
    features.air_quality.shift_pm_2_5(df_aq_update, days=7) # add features about 7 previous PM2.5 values

    features.air_quality.moving_average(df_aq_update, 7)
    features.air_quality.moving_average(df_aq_update, 14)
    features.air_quality.moving_average(df_aq_update, 28)

    for i in [7, 14, 28]:
        for func in [features.air_quality.moving_std,
                     features.air_quality.exponential_moving_average,
                     features.air_quality.exponential_moving_std
                     ]:
            func(df_aq_update, i)


    df_aq_update = df_aq_update.sort_values(by=["date", "pm2_5"]).dropna()
    df_aq_update = df_aq_update.reset_index(drop=True)


    features.air_quality.year(df_aq_update)
    features.air_quality.day_of_month(df_aq_update)
    features.air_quality.month(df_aq_update)
    features.air_quality.day_of_week(df_aq_update)
    features.air_quality.is_weekend(df_aq_update)
    features.air_quality.sin_day_of_year(df_aq_update)
    features.air_quality.cos_day_of_year(df_aq_update)
    features.air_quality.sin_day_of_week(df_aq_update)
    features.air_quality.cos_day_of_week(df_aq_update)
    
    st.write(df_aq_update.groupby("city_name").max().tail(7))
    st.write("✅ Success!")
    ###

    st.write(3 * "-")
    print_fancy_header('\n🌤📆  Parsing Weather data')

    df_weather_update = parse_weather(last_dates_dict, today)
    st.write(df_weather_update.groupby("city_name").max().tail(7))
    st.write("✅ Successfully parsed!")

    df_aq_update.date = df_aq_update.date.astype(str)
    df_weather_update.date = df_weather_update.date.astype(str)
    
    st.write("\n")
    if st.button('📡 Upload the new data to Hopsworks Feature Store'):
        st.write("Connecting to feature groups...")
        air_quality_fg = fs.get_or_create_feature_group(
            name = 'air_quality',
            version = 1
        )
        weather_fg = fs.get_or_create_feature_group(
            name = 'weather',
            version = 1
        )

        df_aq_update.date = pd.to_datetime(df_aq_update.date)
        df_weather_update.date = pd.to_datetime(df_weather_update.date)

        df_aq_update["unix_time"] = df_aq_update["date"].apply(convert_date_to_unix)
        df_weather_update["unix_time"] = df_weather_update["date"].apply(convert_date_to_unix)

        df_aq_update.date = df_aq_update.date.astype(str)
        df_weather_update.date = df_weather_update.date.astype(str)

        air_quality_fg.insert(df_aq_update,
                              write_options={'wait_for_job': False})
        st.write("Created job to insert parsed PM2.5 data into FS...")
        print("Inserting into air_quality fg.")

        weather_fg.insert(df_weather_update,
                          write_options={'wait_for_job': False})
        st.write("Created job to insert parsed weather data into FS...")
        print("Inserting into weather fg.")
            

st.write(3 * '-')
st.write("\n")
print_fancy_header(text="🖍 Select the cities using the form below. \
                         Click the 'Submit' button at the bottom of the form to continue.",
                   font_size=22)
dict_for_streamlit = {}
for continent in target_cities:
        for city_name, coords in target_cities[continent].items():
            dict_for_streamlit[city_name] = coords
selected_cities_full_list = []

with st.form(key="user_inputs"):
    print_fancy_header(text='\n🗺 Here you can choose cities from the drop-down menu',
                       font_size=20, color="#00FFFF")
    
    cities_multiselect = st.multiselect(label='',
                                        options=dict_for_streamlit.keys())
    selected_cities_full_list.extend(cities_multiselect)
    st.write("_" * 3)
    print_fancy_header(text="\n📌 To add a city using the interactive map, click somewhere \
                             (for the coordinates to appear)",
                       font_size=20, color="#00FFFF")
    
    my_map = folium.Map(location=[42.57, -44.092], zoom_start=2)
    # Add markers for each city
    for city_name, coords in dict_for_streamlit.items():
        folium.CircleMarker(
            location=coords
        ).add_to(my_map)

    my_map.add_child(folium.LatLngPopup())
    res_map = st_folium(my_map, width=640, height=480)
    
    try:
        new_lat, new_long = res_map["last_clicked"]["lat"], res_map["last_clicked"]["lng"]


        # Calculate the distance between the clicked location and each city
        distances = {city: distance.distance(coord, (new_lat, new_long)).km for city, coord in dict_for_streamlit.items()}

        # Find the city with the minimum distance and print its name
        nearest_city = min(distances, key=distances.get)
        print_fancy_header(text=f"You have selected {nearest_city} using map", font_size=18, color="#52fa23")
        
        selected_cities_full_list.append(nearest_city)
        # st.write(label_encoder.transform([nearest_city])[0])

    except Exception as err:
        print(err)
        pass
    
    print_fancy_header(text='\n🧮 How many days do you want me to predict?',
                 font_size=18, color="#00FFFF")
    options = [3, 7, 10, 14]
    
#     st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

#     st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{padding-left:2px;}</style>',
#              unsafe_allow_html=True)

    HOW_MANY_DAYS_PREDICT = st.radio("", options, index=1)

    HOW_MANY_DAYS_PREDICT = int(HOW_MANY_DAYS_PREDICT)

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    st.write('Selected cities:', selected_cities_full_list)

    st.write(3*'-')
    if str(today) == max(last_dates_dict.values()):
        dataset = batch_data
    else:
        # batch_data.to_csv("debug/batch_data.csv", index=False)
        # df_weather_update.to_csv("debug/df_weather_update.csv", index=False)
        # df_aq_update.to_csv("debug/df_aq_update.csv", index=False)
        
        updates = df_weather_update.merge(df_aq_update, on=['city_name', 'date'])
        dataset = pd.concat([batch_data, updates]).drop(columns=['unix_time']).reset_index(drop=True)
        
        # dataset.to_csv("debug/dataset.csv", index=False)
    batch_data.to_csv("debug/batch_data.csv", index=False)
    
    dataset = dataset.drop_duplicates(subset=['city_name', 'date'])
    dataset = dataset.sort_values(by=["city_name", "date"])

    saved_model_dir = download_model(
        name="air_quality_xgboost_model",
        version=1
    )

    retrieved_xgboost_model = joblib.load(saved_model_dir + "/xgboost_regressor.pkl")
    retrieved_encoder = joblib.load(saved_model_dir + "/label_encoder.pkl")

    print_fancy_header("\n🧬 Modeling")
    st.write("\n")
    print_fancy_header(text='\n🤖 Getting the model...',
                       font_size=18, color="#FDF4F5")


    saved_model_dir = download_model(
        name="air_quality_xgboost_model",
        version=1
    )
    st.write("\n")
    st.write("✅ Model was downloaded and cached.")
    
    regressor = joblib.load(saved_model_dir + "/xgboost_regressor.pkl")
    encoder = joblib.load(saved_model_dir + "/label_encoder.pkl")

    print_fancy_header(text='\n🧠 Predicting...',
                       font_size=18, color="#FDF4F5")
    
    
    for city_name in selected_cities_full_list:
        st.write(f"Processing {city_name}...")
        temp_date = datetime.date.today()
        for i in range(HOW_MANY_DAYS_PREDICT):
            temp_date += datetime.timedelta(days=1)

            df_aq_temp = pd.DataFrame(columns=dataset.columns, data=[[-1] * dataset.shape[1]])
            df_aq_temp['date'] = temp_date
            df_aq_temp['city_name'] = city_name
            
            df_aq_temp = pd.concat([dataset, df_aq_temp], axis=0).reset_index(drop=True)

            df_aq_temp['date'] = pd.to_datetime(df_aq_temp['date'])
            features.air_quality.shift_pm_2_5(df_aq_temp, days=7) # add features about 7 previous PM2.5 values

            features.air_quality.moving_average(df_aq_temp, 7)
            features.air_quality.moving_average(df_aq_temp, 14)
            features.air_quality.moving_average(df_aq_temp, 28)

            for i in [7, 14, 28]:
                for func in [features.air_quality.moving_std,
                             features.air_quality.exponential_moving_average,
                             features.air_quality.exponential_moving_std
                             ]:
                    func(df_aq_temp, i)

            df_aq_temp = df_aq_temp.sort_values(by=["date", "pm2_5"]).dropna()
            df_aq_temp = df_aq_temp.reset_index(drop=True)

            features.air_quality.year(df_aq_temp)
            features.air_quality.day_of_month(df_aq_temp)
            features.air_quality.month(df_aq_temp)
            features.air_quality.day_of_week(df_aq_temp)
            features.air_quality.is_weekend(df_aq_temp)
            features.air_quality.sin_day_of_year(df_aq_temp)
            features.air_quality.cos_day_of_year(df_aq_temp)
            features.air_quality.sin_day_of_week(df_aq_temp)
            features.air_quality.cos_day_of_week(df_aq_temp)

            # we need only the last row (one city, one day)
            df_aq_temp = df_aq_temp[df_aq_temp['city_name'] == city_name].tail(1)

            # get weather data for this specific day
            coordinates = dict_for_streamlit[city_name]
            df_weather_temp = get_weather_data_from_open_meteo(city_name=city_name,
                                                               coordinates=coords,
                                                               start_date=str(temp_date),
                                                               end_date=str(temp_date),
                                                               forecast=True)

            df_aq_temp = df_aq_temp.drop(columns=df_weather_temp.columns[2:])

            X = df_aq_temp.merge(df_weather_temp, on=["city_name", "date"])
            encoded = encoder.transform(X['city_name'])

            # Convert the output to a dense array and concatenate with the original data
            X = pd.concat([X, pd.DataFrame(encoded)], axis=1)
            X = X.rename(columns={0: 'city_name_encoded'})

            feature_names = regressor.get_booster().feature_names
            X = X[feature_names]

            preds_temp = regressor.predict(X)

            df_temp = X.copy()
            df_temp['pm2_5'] = round(preds_temp[0], 1)
            df_temp['city_name'] = city_name
            df_temp['date'] = str(temp_date)
            df_temp = df_temp.drop(columns=['city_name_encoded'])

            # update dataset variable
            dataset = pd.concat([dataset, df_temp])



    plot_pm2_5(dataset[dataset['city_name'].isin(selected_cities_full_list)])
    ######################## MODELING MODELING MODELING MODELING

    st.write(36 * "-")
    st.subheader('\n🎉 📈 🤝 App Finished Successfully 🤝 📈 🎉')
    st.button("Re-run")
