"""
Streamlit app for Glocal Aggregations

App created by: Shreyas Gadgin Matha
"""

import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import gcsfs


# Set app configurations
st.set_page_config(
    page_title="Glocal Aggregations", page_icon=":earth_asia:", layout="wide"
)


def create_gcp_client():
    # Create GCP client
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = storage.Client(credentials=credentials)
    return client


def prepare_gcsfs():
    # Get GCP client
    client = create_gcp_client()
    # Create GCSFS
    fs = gcsfs.GCSFileSystem(project=client.project, token=client._credentials)
    return fs


def gcsfs_to_pandas(fs, BUCKET_NAME, file_name, columns=None):
    with fs.open(f"{BUCKET_NAME}/{file_name}") as f:
        if file_name.endswith(".parquet"):
            df = pd.read_parquet(f, columns=columns)
        elif file_name.endswith(".csv"):
            if columns is not None:
                raise ValueError("Columns not supported for CSV files")
            df = pd.read_csv(f)
        else:
            raise ValueError("File format not supported")
    return df


def gcsfs_to_geopandas(fs, BUCKET_NAME, file_name, columns=None):
    with fs.open(f"{BUCKET_NAME}/{file_name}") as f:
        if file_name.endswith(".parquet"):
            df = gpd.read_parquet(f, columns=columns)
        elif file_name.endswith(".shp"):
            if columns is not None:
                raise ValueError("Columns not supported for Shapefiles")
            df = gpd.read_file(f)
        else:
            raise ValueError("File format not supported")
    return df


# List of datasets being read
# 1. Country codes
# 2. Annualized aggregations data - detailed level
# 3. Variable rank - country level
# 4. Variable missings - country level
# 5. Shapefiles - detailed level


@st.experimental_memo(ttl=900)
def read_data(path_in_bucket, columns=None, spatial=False):
    # Get GCSFS
    fs = prepare_gcsfs()
    # Set GCS bucket name
    BUCKET_NAME = "glocal"
    if not spatial:
        df = gcsfs_to_pandas(fs, BUCKET_NAME, path_in_bucket, columns=columns)
    else:
        df = gcsfs_to_geopandas(fs, BUCKET_NAME, path_in_bucket, columns=columns)
    return df


# -------------------------#
# Set up sidebar
# -------------------------#
st.sidebar.title("Viz parameters")
# Read general data
country_codes = read_data("country_codes.parquet")
# Country
selected_country_name = st.sidebar.selectbox(
    "Country", country_codes.country_name.unique()
)
selected_country = country_codes[
    country_codes.country_name == selected_country_name
].country_code.values[0]
# Variable
available_cols = read_data("available_cols.parquet")
varlist = [x for x in available_cols.colname if x not in ["year", "GID_0"]]
selected_var = st.sidebar.selectbox("Variable", varlist)
# GADM level
selected_gadm_string = st.sidebar.radio("GADM level", ["GID_0", "GID_1", "GID_2"])
selected_gadm_level = int(selected_gadm_string[-1])


# ------------------------------------
# Data reading functions
# ------------------------------------
# Read aggregations
@st.experimental_memo(ttl=900)
def read_glocal_var(level, selected_var):
    if level == 0:
        vars_to_read = ["year", "GID_0", selected_var]
    elif level == 1:
        vars_to_read = ["year", "GID_0", "GID_1", selected_var]
    elif level == 2:
        vars_to_read = ["year", "GID_0", "GID_2", selected_var]
    else:
        raise ValueError("GADM level not supported")
    df = read_data(
        f"annualized_level_{level}.parquet",
        columns=vars_to_read,
    )
    df = df.dropna(subset=[selected_var])
    return df


# Read glocal data
glocal_0 = read_glocal_var(0, selected_var)

# Ranks
glocal_0_rank = read_data(
    "supporting_data/glocal_0_rank.parquet", columns=["year", "GID_0", selected_var]
)

# Missing values
glocal_missing_dict = {}
for x in [0, 1, 2]:
    glocal_missing_dict[x] = read_data(
        f"supporting_data/glocal_{x}_missing.parquet",
        columns=["year", "GID_0", selected_var],
    )

# Get the latest year for which data is available for the selected country at each level
availability_dict = {}
for x in [0, 1, 2]:
    missingvals_year = glocal_missing_dict[x].loc[
        (glocal_missing_dict[x]["GID_0"] == selected_country)
        & (glocal_missing_dict[x][selected_var] < 1),
        "year",
    ]
    availability_dict[x] = (missingvals_year.min(), missingvals_year.max())


# ----------------
# Add additional elements to sidebar
# Multiselect for comparator countries
selected_comparator_names = st.sidebar.multiselect(
    label="Select comparator countries",
    options=[
        x for x in country_codes.country_name.unique() if x != selected_country_name
    ],
    default=None,
)
if selected_comparator_names:
    selected_comparators = country_codes[
        country_codes.country_name.isin(selected_comparator_names)
    ].country_code.values
    selected_countries = [selected_country] + list(selected_comparators)
else:
    selected_countries = [selected_country]

# Slider select for year
selected_year = st.sidebar.slider(
    "Select years for analysis",
    min_value=availability_dict[selected_gadm_level][0],
    max_value=availability_dict[selected_gadm_level][1],
    value=availability_dict[selected_gadm_level],
    step=1,
)


# ------------------------------------
# Intro text
# ------------------------------------

st.title("Glocal Aggregations")
st.markdown(
    """
    This app visualizes the aggregations developed as part of the Glocal project, which aims to develop a dataset that is globally comparable and yet granular enough to be locally relevant. The aggregations are developed at three levels: GID_0 (country), GID_1 (state/province), and GID_2 (county), using GADM v3.6.

    Resources:
    - [Codebook](https://docs.google.com/spreadsheets/d/1JWInGw6vcGZPi3TgEsZ66OyCA_t_kLjGYhIZTIJyl_Q/edit#gid=0)
    - [Github Repository](https://github.com/cid-harvard/glocal_aggregations)
    - [Data](https://www.dropbox.com/sh/tphr6wkxhggzgke/AAB062OWCrvoYjg6gQ8pjFgPa?dl=0)
    """
)

# ------------------------------------
# National level exhibits
# ------------------------------------
st.markdown(
    f"""
    ## National Level Exhibits

    ### Data availability

    Earliest and latest year for which data is available at each level:
    - Level 0: {availability_dict[0]}
    - Level 1: {availability_dict[1]}
    - Level 2: {availability_dict[2]}
    """
)

# ----------------
# Plot the missing value percentage of the selected country for the selected variable

# Filter data
missing_val_df = glocal_missing_dict[selected_gadm_level]
missing_year = missing_val_df.loc[
    (missing_val_df["GID_0"].isin(selected_countries))
    & (missing_val_df.year.between(*selected_year)),
    ["year", "GID_0", selected_var],
]
# Lineplot
missing_px = px.line(
    missing_year,
    x="year",
    y=selected_var,
    color="GID_0",
    title="Fraction of values missing",
    markers=True,
    labels={"year": "Year", selected_var: "Fraction of values missing"},
)
missing_px.update_xaxes(tickformat="%Y")
st.plotly_chart(missing_px, use_container_width=True)

# ----------------
# # Get the latest percentile rank of the selected country for the selected variable
# rank = glocal_0_rank.loc[
#     (glocal_0_rank["GID_0"].isin(selected_countries))
#     & (glocal_0_rank["year"] == availability_dict[0][1]),
#     selected_var,
# ].values[0]

# ----------------
# Plot the time series of the selected country for the selected variable
# Filter data
var_year = glocal_0.loc[
    (glocal_0["GID_0"].isin(selected_countries))
    & (glocal_0.year.between(*selected_year)),
    ["year", "GID_0", selected_var],
]
# Lineplot
var_year_px = px.line(
    var_year,
    x="year",
    y=selected_var,
    color="GID_0",
    title=f"Time series of {selected_var}",
    markers=True,
)
var_year_px.update_xaxes(tickformat="%Y")
st.plotly_chart(var_year_px, use_container_width=True)

# ----------------
# Plot the time series of the rank for the selected country for the selected variable
# Filter data
var_rank_year = glocal_0_rank.loc[
    (glocal_0_rank["GID_0"].isin(selected_countries))
    & (glocal_0_rank.year.between(*selected_year)),
    ["year", "GID_0", selected_var],
].dropna()
# Lineplot
var_rank_year_px = px.line(
    var_rank_year,
    x="year",
    y=selected_var,
    color="GID_0",
    title=f"Rank for variable {selected_var}",
    markers=True,
    labels={"x": "Year", "y": f"Rank for {selected_var}"},
)
var_rank_year_px.update_xaxes(tickformat="%Y")
st.plotly_chart(var_rank_year_px, use_container_width=True)


# ------------------------------------
# Subnational level exhibits
# ------------------------------------

# Get level
if selected_gadm_level in [0, 1]:
    subnational_gadm_level = 1
elif selected_gadm_level == 2:
    subnational_gadm_level = 2
else:
    raise ValueError("GADM level must be 0, 1, or 2.")


# Choropleth map
st.markdown(
    f"""
    ## Subnational trends

    Subnational trends for {selected_var} averaged over the years: {selected_year[0]}-{selected_year[1]}, at GADM level {subnational_gadm_level} administrative boundaries.
    """
)
# Read glocal data
if subnational_gadm_level == 1:
    glocal = read_glocal_var(1, selected_var)
    glocal = glocal[(glocal["GID_0"] == selected_country)]
elif selected_gadm_level == 2:
    glocal = read_glocal_var(2, selected_var)
    glocal = glocal[(glocal["GID_0"] == selected_country)]
else:
    raise ValueError("Subnational GADM level must be 1, or 2.")

# Add a year filter
glocal_subnational = (
    glocal[glocal["year"].between(*selected_year)]
    .groupby(f"GID_{subnational_gadm_level}")[selected_var]
    .mean()
    .reset_index()
)

# Read in shapefile and convert to geojson
@st.experimental_memo(ttl=900)
def get_country_shapefile(level, country):
    vars_to_read = [
        f"GID_{level}",
        f"NAME_{level}",
        "geometry",
    ]
    gdf = read_data(
        f"simplified_shapefiles/gadm/country_level/gadm_{level}/{country}.parquet",
        columns=vars_to_read,
        spatial=True,
    )
    gdf_json = json.loads(gdf.to_json())
    return gdf_json


# Read country shapefile
gdf_json = get_country_shapefile(subnational_gadm_level, selected_country)

# Plotly choropleth map
chropleth_px = px.choropleth(
    glocal_subnational,
    geojson=gdf_json,
    locations=f"GID_{subnational_gadm_level}",
    featureidkey=f"properties.GID_{subnational_gadm_level}",
    color=selected_var,
    color_continuous_scale="Viridis",
    labels={
        selected_var: selected_var,
        f"GID_{subnational_gadm_level}": "GID",
        f"NAME_{subnational_gadm_level}": "Region",
    },
)
chropleth_px.update_geos(fitbounds="locations", visible=True)
chropleth_px.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(chropleth_px, use_container_width=True)
