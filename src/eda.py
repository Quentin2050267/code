import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
from geopy.geocoders import Nominatim
import io

def get_description(data):
    st.subheader("Original Data Overview")
    st.dataframe(
        data,
        height=500,
        width=900,
        )

    data_desc = data.describe()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Temperature", f"{data_desc.loc["mean", "temperature"]:.2f}")
    col2.metric("Avg Pressure", f"{data_desc.loc["mean", "pressure"]:.2f}")
    col3.metric("Avg Humidity", f"{data_desc.loc["mean", "humidity"]:.2f}")
    col4.metric("Avg Vibration", f"{data_desc.loc["mean", "vibration"]:.2f}")

    with st.expander("Show Detailed Descriptive Statistics"):
        st.dataframe(data_desc.T)
        st.caption("Vibration has negative values, suggesting potential outliers.")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

def get_eda_insights(data):
    st.subheader("Exploratory Data Analysis")
    st.markdown("This section will show the detailed data analysis results.")
    st.write("#### Equipment Location on Map")

    # # can use geopy to get the latitude and longitude of each location
    # # but here we use a pre-defined dictionary to save time
    # geolocator = Nominatim(user_agent="geo_locator")
    # def get_lat_lon(city):
    #     try:
    #         location = geolocator.geocode(city)
    #         return (location.latitude, location.longitude)
    #     except:
    #         return (None, None)
    
    # locations = data["location"].unique()
    # location_dict = {location: get_lat_lon(location) for location in locations}

    location_dict = {
        "New York": (40.7127281, -74.0060152),
        "San Francisco": (37.7792588, -122.4193286),
        "Atlanta": (33.7489924, -84.3902644),
        "Chicago": (41.8755616, -87.6244212),
        "Houston": (29.7589382, -95.3676974),
    }
    location_df = pd.DataFrame(location_dict).T.reset_index()
    location_df.columns = ["location", "latitude", "longitude"]
    data = data.merge(location_df, on="location", how="left")


    city_data = data.groupby(["location"], as_index=False).agg({
        "pressure": lambda x: round(x.mean(), 2),
        "temperature": lambda x: round(x.mean(), 2),
        "humidity": lambda x: round(x.mean(), 2),
        "vibration": lambda x: round(x.mean(), 2),
        "latitude": "first",
        "longitude": "first",
        "faulty": "count",
    })

    city_data["size"] = city_data["faulty"] * 100


    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=37.0902,
                longitude=-95.7129,
                zoom=2.8,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=city_data,
                    get_position=["longitude", "latitude"],
                    elevation_scale=400,
                    elevation_range=[0, 2000],
                    auto_highlight=True,
                    pickable=True,
                    extruded=True,
                    get_radius="size",
                    get_color="[255, 75, 75]",
                )
            ],
            tooltip={
                "html": """
                    <b>City:</b> {location}<br>
                    <b>Avg Pressure:</b> {pressure}<br>
                    <b>Avg Temperature:</b> {temperature}<br>
                    <b>Avg Humidity:</b> {humidity}<br>
                    <b>Avg Vibration:</b> {vibration}<br>
                    <b>Total:</b> {faulty}
                """
            },
        )
    )
    st.caption(f"This dataset contains {data['location'].nunique()} unique locations, which are {data['location'].unique()}.")

    numeric_data = data.select_dtypes(include=[np.number]).drop(columns=["latitude", "longitude"])
    category_data = data.select_dtypes(include=[object])

    st.write("### Missing Value Bar Chart")
    missing_values = data.isnull().sum()
    missing_df = pd.DataFrame({"Column": missing_values.index, "Missing Count": missing_values.values})

    st.bar_chart(missing_df.set_index("Column"))
    st.caption("Bar chart showing missing values count per column. No missing values detected, confirming data completeness for further processing.")


    st.write("#### Correlation Matrix")
    corr_matrix = numeric_data.corr()
    st.vega_lite_chart(corr_matrix.reset_index().melt(id_vars='index'), {
        'mark': 'rect',
        'width': 400,
        'height': 650,
        'encoding': {
            'x': {'field': 'index', 'type': 'nominal'},
            'y': {'field': 'variable', 'type': 'nominal'},
            'color': {'field': 'value', 'type': 'quantitative'}
        }
    })
    st.caption("Mostly no significant correlation between numeric features, except for 'faulty' and 'vibration', 'temperature' and 'pressure' which show a certain relationship, guiding feature prioritization.")


    # st.write("#### Pair Plot of Numeric Features")
    # pair_fig = sns.pairplot(numeric_data, hue="faulty", diag_kind="hist")
    # st.pyplot(pair_fig.figure)


    # Scatter plots for numeric features
    st.write("#### Scatter Plots for Numeric Features")
    scatter_col1, scatter_col2, scatter_col3 = st.columns(3)
    with scatter_col1:
        st.scatter_chart(data, x="temperature", y="humidity", size=5, color=None)
        st.scatter_chart(data, x="temperature", y="pressure", size=5, color=None)
    with scatter_col2:
        st.scatter_chart(data, x="temperature", y="vibration", size=5, color=None)
        st.scatter_chart(data, x="pressure", y="vibration", size=5, color=None)
    with scatter_col3:
        st.scatter_chart(data, x="humidity", y="vibration", size=5, color=None)
        st.scatter_chart(data, x="pressure", y="humidity", size=5, color=None)
    st.caption("Scatter plots for exploring correlations between numeric features, no clear pattern observed, further confirming no significant correlation.") 
    
    st.write("#### Data Distribution")
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(data=numeric_data, x="pressure", kde=False, ax=ax[0, 0], hue="faulty")
    sns.histplot(data=numeric_data, x="temperature", kde=False, ax=ax[0, 1], hue="faulty")
    sns.histplot(data=numeric_data, x="humidity", kde=False, ax=ax[1, 0], hue="faulty")
    sns.histplot(data=numeric_data, x="vibration", kde=False, ax=ax[1, 1], hue="faulty")
    st.pyplot(fig)
    st.caption("Data distribution for each numeric feature. Humidity tends to follow a normal distribution, while the other three have some skewness, with vibration showing the most significant long-tail effect, suggesting a log transformation. Fault values are more prominent at temperature extremes.")


    pie_col1, pie_col2, pie_col3 = st.columns(3)
    with pie_col1:
        # Equipment distribution
        st.write("#### Equipment Distribution")
        equipment_counts = data['equipment'].value_counts().reset_index()
        equipment_counts.columns = ['equipment', 'count']
        equipment_counts['percentage'] = round((equipment_counts['count'] / equipment_counts['count'].sum()) * 100, 2).astype(str) + "%"
        st.vega_lite_chart(equipment_counts, {
            'mark': 'arc',
            'width': 200,
            'height': 200,
            'encoding': {
                'theta': {'field': 'count', 'type': 'quantitative'},
                'color': {'field': 'equipment', 'type': 'nominal'},
                'tooltip': [{'field': 'equipment', 'type': 'nominal'}, {'field': 'count', 'type': 'quantitative'}, {'field': 'percentage', 'type': 'nominal'}]
            }
        })
        st.caption("Distribution of equipment, generally balanced.")
    with pie_col2:
        # location distribution
        st.write("#### Location Distribution")
        location_counts = data['location'].value_counts().reset_index()
        location_counts.columns = ['location', 'count']
        location_counts['percentage'] = round((location_counts['count'] / location_counts['count'].sum()) * 100, 2).astype(str) + "%"
        st.vega_lite_chart(location_counts, {
            'mark': 'arc',
            'width': 200,
            'height': 200,
            'encoding': {
                'theta': {'field': 'count', 'type': 'quantitative'},
                'color': {'field': 'location', 'type': 'nominal'},
                'tooltip': [{'field': 'location', 'type': 'nominal'}, {'field': 'count', 'type': 'quantitative'}, {'field': 'percentage', 'type': 'nominal'}]
            }
        })
        st.caption("Distribution of locations, generally balanced.")
    with pie_col3:
        # fault distribution
        st.write("#### Fault Distribution")
        faulty_counts = data['faulty'].value_counts().reset_index()
        faulty_counts.columns = ['faulty', 'count']
        faulty_counts['percentage'] = round((faulty_counts['count'] / faulty_counts['count'].sum()) * 100, 2).astype(str) + "%"
        st.vega_lite_chart(faulty_counts, {
            'mark': 'arc',
            'width': 200,
            'height': 200,
            'encoding': {
                'theta': {'field': 'count', 'type': 'quantitative'},
                'color': {'field': 'faulty', 'type': 'nominal'},
                'tooltip': [{'field': 'faulty', 'type': 'nominal'}, {'field': 'count', 'type': 'quantitative'}, {'field': 'percentage', 'type': 'nominal'}]
            }
        })
        st.caption("Distribution of faulty equipment, unbalanced, more non-faulty equipment, suggesting need for balancing techniques.")


    bar_col1, bar_col2, bar_col3 = st.columns(3)
    with bar_col1:
        # Distribution of equipment across different locations.
        st.write("#### Equipment Location Distribution")
        equipment_location_counts = data.groupby(['equipment', 'location']).size().reset_index()
        equipment_location_counts.columns = ['equipment', 'location', 'count']
        equipment_location_counts_pivot = equipment_location_counts.pivot(index='location', columns='equipment', values='count').fillna(0)
        st.bar_chart(equipment_location_counts_pivot)
        st.caption("Distribution of equipment across different locations, even in different locations, the distribution is generally balanced.")
    with bar_col2:
        # Distribution of faulty across different locations.
        st.write("#### Fault Location Distribution")
        faulty_location_counts = data.groupby(['faulty', 'location']).size().reset_index()
        faulty_location_counts.columns = ['faulty', 'location', 'count']
        faulty_location_counts_pivot = faulty_location_counts.pivot(index='location', columns='faulty', values='count').fillna(0)
        st.bar_chart(faulty_location_counts_pivot)
        st.caption("Distribution of faulty equipment across different locations, generally balanced.")
    with bar_col3:
        # Distribution of faulty across different equipment.
        st.write("#### Fault Equipment Distribution")
        faulty_equipment_counts = data.groupby(['faulty', 'equipment']).size().reset_index()
        faulty_equipment_counts.columns = ['faulty', 'equipment', 'count']
        faulty_equipment_counts_pivot = faulty_equipment_counts.pivot(index='equipment', columns='faulty', values='count').fillna(0)
        st.bar_chart(faulty_equipment_counts_pivot)
        st.caption("Distribution of faulty equipment across different equipment, generally balanced.")

    # 计算故障率并排序
    location_fault_rate = data.groupby("location")["faulty"].mean().reset_index().set_index("location").sort_values(by="faulty", ascending=True)
    equipment_fault_rate = data.groupby("equipment")["faulty"].mean().reset_index().set_index("equipment").sort_values(by="faulty", ascending=True)

    st.write("#### Fault Rate by Location")
    st.bar_chart(location_fault_rate, horizontal=True, use_container_width=True)
    location_sort_caption = f"{location_fault_rate.index[0]} has the lowest fault rate, while {location_fault_rate.index[-1]} has the highest fault rate. But the difference is not significant."
    st.caption(f"Fault rate by location. {location_sort_caption}")
    st.write("#### Fault Rate by Equipment")
    st.bar_chart(equipment_fault_rate, horizontal=True)
    st.caption("Fault rate by equipment, all equipment have similar fault rates, with no significant difference.")


    # kde plots
    st.write("#### Distribution by Equipment")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(["temperature", "pressure", "vibration", "humidity"]):
        sns.kdeplot(data=data, x=col, ax=axs[i // 2, i % 2], hue="equipment", fill=False, alpha=1, palette="Paired")
    st.pyplot(fig)
    st.caption("The distribution of 4 features for each equipment. The distributions of different equipment are roughly similar, with some slight differences. For temperature and pressure, the Turbine and Compressor have similar distributions, but the Pump has a slightly lower peak, meaning its values are more spread out. For vibration, the three curves largely overlap, indicating similar variation across equipment types. For humidity, there are some fluctuations at the peak and in the tail regions, suggesting slight differences in distribution among equipment types.")
    
    st.write("#### Distribution by Location")
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(["temperature", "pressure", "vibration", "humidity"]):
        sns.kdeplot(data=data, x=col, ax=axs[i // 2, i % 2], hue="location", fill=False, alpha=1, palette="Paired")
    st.pyplot(fig)
    st.caption("The distribution of 4 features for each location. The distributions of different equipment are roughly similar, with some slight differences at the peak and tail regions, among which humidity values vary more across locations than other features.")

    container = st.container(border=True)
    with container:
        st.write("Based on the above exploratory data analysis, I get some insights that will help me during the data preprocessing:")
        st.write("1. The data is complete, with no missing values.")
        st.write("2. The outlier detection is needed for 'vibration' due to negative values.")
        st.write("2. The correlation between numeric features is generally low, except for 'faulty' and 'vibration'.")
        st.write("3. The distribution of numeric features shows some skewness, especially for 'vibration', suggesting a log transformation.")
        st.write("4. The distribution of 'faulty' is imbalanced, with more non-faulty equipment, suggesting the need for balancing techniques.")
        st.write("5. The distribution of equipment and location is generally balanced.")
        st.write("6. The fault rate by location and equipment is similar, with no significant difference.")

    fault_counts = data['faulty'].value_counts()
    return {
        'outliers': {'vibration': data['vibration'].min() < 0},
        'skewed_features': ['vibration'],
        'imbalanced': fault_counts[1] / fault_counts.sum() < 0.2,
        'key_features': ['vibration']  # based on correlation
    }