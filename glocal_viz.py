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
            try:
                df = pd.read_parquet(f, columns=columns)
            except Exception as e:
                st.error(f"Error reading parquet file {file_name}")
                raise e
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


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


# List of datasets being read
# 1. Country codes
# 2. Annualized aggregations data - detailed level
# 3. Variable rank - country level
# 4. Variable missings - country level
# 5. Shapefiles - detailed level


@st.cache_data(ttl=900)
def read_data(path_in_bucket, columns=None, spatial=False):
    # Get GCSFS
    fs = prepare_gcsfs()
    # Set GCS bucket name
    BUCKET_NAME = "glocal_streamlit"
    if not spatial:
        df = gcsfs_to_pandas(fs, BUCKET_NAME, path_in_bucket, columns=columns)
    else:
        df = gcsfs_to_geopandas(fs, BUCKET_NAME, path_in_bucket, columns=columns)
    return df


# Read in shapefile just for specific non-spatial variables
@st.cache_data(ttl=900)
def get_country_shapefile(level, country):
    vars_to_read = [
        f"GID_{level}",
        f"NAME_{level}",
    ]
    gdf = read_data(
        f"simplified_shapefiles/gadm/country_level/gadm_{level}/{country}.parquet",
        columns=vars_to_read,
        spatial=False,
    )
    # Ensure that all columns are present
    for col in vars_to_read:
        if col not in gdf.columns:
            st.error(f"Column {col} not found in shapefile")
            raise ValueError(f"Column {col} not found in shapefile")
    return gdf


# Read in shapefile and convert to geojson
@st.cache_data(ttl=900)
def get_country_shapefile_as_geojson(level, country):
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
    # Ensure that all columns are present
    for col in vars_to_read:
        if col not in gdf.columns:
            st.error(f"Column {col} not found in shapefile")
            raise ValueError(f"Column {col} not found in shapefile")

    gdf_json = json.loads(gdf.to_json())

    return gdf_json


# -------------------------#
# Set up sidebar
# -------------------------#
st.sidebar.title("Viz parameters")

# Read general data

country_codes = read_data("country_codes.parquet")
docs = read_data("docs.parquet")
available_cols = read_data("available_cols.parquet")

# Country selection
selected_country_name = st.sidebar.selectbox(
    "Country", country_codes.country_name.unique(), help="Select a country to analyze"
)
selected_country = country_codes[
    country_codes.country_name == selected_country_name
].country_code.values[0]

# Variable selection
varlist = list(available_cols.variable_name)
# Remove "Country GID" and "Year" from the list
varlist.remove("Country GID")
varlist.remove("Year")
selected_var_name = st.sidebar.selectbox(
    "Variable", varlist, help="Select a variable to visualize"
)
selected_var = available_cols[
    available_cols.variable_name == selected_var_name
].colname.values[0]

# GADM level selection
selected_gadm_string = st.sidebar.radio(
    "GADM level",
    ["GID_0", "GID_1", "GID_2"],
    help="Select the administrative level for analysis",
)
selected_gadm_level = int(selected_gadm_string[-1])

# Comparator countries selection
selected_comparator_names = st.sidebar.multiselect(
    label="Select Comparator Countries",
    options=[
        x for x in country_codes.country_name.unique() if x != selected_country_name
    ],
    default=None,
    help="Select comparator countries for analysis (optional)",
)
if selected_comparator_names:
    selected_comparators = country_codes[
        country_codes.country_name.isin(selected_comparator_names)
    ].country_code.values
    selected_countries = [selected_country] + list(selected_comparators)
else:
    selected_countries = [selected_country]


# ------------------------------------
# Data reading functions
# ------------------------------------
# Read aggregations
@st.cache_data(ttl=900)
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
    "glocal_0_rank.parquet", columns=["year", "GID_0", selected_var]
)

# Missing values
glocal_missing_dict = {}
for x in [0, 1, 2]:
    glocal_missing_dict[x] = read_data(
        f"glocal_{x}_missing.parquet",
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
    if missingvals_year.min() == missingvals_year.max():
        availability_dict[x] = (missingvals_year.min(), missingvals_year.max() + 1)
    else:
        availability_dict[x] = (missingvals_year.min(), missingvals_year.max())


# ----------------
# Add additional elements to sidebar
# Slider select for year
selected_year = st.sidebar.slider(
    "Select years for analysis",
    min_value=int(availability_dict[selected_gadm_level][0]),
    max_value=int(availability_dict[selected_gadm_level][1]),
    value=tuple([int(x) for x in availability_dict[selected_gadm_level]]),
    step=1,
)


# ------------------------------------
# Intro text
# ------------------------------------

with open("style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

st.title("Glocal Aggregations")
st.markdown(
    """
    This app visualizes the aggregations developed as part of the Glocal project, which aims to develop a dataset that is globally comparable and yet granular enough to be locally relevant. The aggregations are developed at three levels of administrative boundaries: GID_0 (country), GID_1 (state/province), and GID_2 (county), using boundaries data from [GADM v3.6](https://gadm.org/download_world36.html).

    Key features:
    - Explore a wide range of economic, demographic, ecological, and socio-political variables
    - Compare data across countries and subnational regions
    - Visualize trends and rankings over time
    - Access detailed variable information and documentation

    Resources:
    - [Data Download](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6TUCTE)
    - [Codebook](https://docs.google.com/spreadsheets/d/1fpoI3AFh821tEuVSOwXm86kXWU7c4lt8J1-fnhysUn0/edit?usp=sharing)
    - [Github Repository](https://github.com/shreyasgm/glocal)
    """
)

docs_dict = docs.loc[docs.variable == selected_var].to_dict(orient="records")[0]

st.markdown(
    f"""
    ## Variable Information

    |||
    |-------|-------|
    | Variable | {docs_dict["variable_name"]} |
    | Units | {docs_dict["units"]} |
    | Description | {docs_dict["description"]} |
    | Frequency | {docs_dict["frequency"]} |
    | Resolution | {docs_dict["resolution"]} |
    | Dataset Name | {docs_dict["dataset_name"]} |
    | Source | {docs_dict["source"]} |
    | Source URL | {docs_dict["source_url"]} |
    | Terms of Use | {docs_dict["license_terms_of_use"]} |
    | Citation | {docs_dict["citation"]} |
    
    """
)

# ------------------------------------
# National level exhibits
# ------------------------------------
# Data availability
st.markdown(
    f"""
    ## Data Availability

    | GADM Level | Earliest Year | Latest Year |
    |------------|---------------|-------------|
    | Level 0    | {availability_dict[0][0]} | {availability_dict[0][1] - 1} |
    | Level 1    | {availability_dict[1][0]} | {availability_dict[1][1] - 1} |
    | Level 2    | {availability_dict[2][0]} | {availability_dict[2][1] - 1} |
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
    template="plotly_white",
)
missing_px.update_xaxes(tickformat="%Y")
missing_px.update_layout(
    legend_title="Country",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(missing_px, use_container_width=True)

# ----------------
# # Get the latest percentile rank of the selected country for the selected variable
# rank = glocal_0_rank.loc[
#     (glocal_0_rank["GID_0"].isin(selected_countries))
#     & (glocal_0_rank["year"] == availability_dict[0][1]),
#     selected_var,
# ].values[0]

# ----------------
st.markdown("## National Trends")

if missingvals_year.min() == missingvals_year.max():
    st.markdown(
        f"""
        Trend data unavailable. Data is available for the selected variable for the selected country only for the year {missingvals_year.min()}.
        """
    )

# Plot the time series of the selected country for the selected variable
# Filter data
var_year = glocal_0.loc[
    (glocal_0["GID_0"].isin(selected_countries))
    & (glocal_0.year.between(*selected_year)),
    ["year", "GID_0", selected_var],
].sort_values(["year", "GID_0"])
# Lineplot
var_year_px = px.line(
    var_year,
    x="year",
    y=selected_var,
    color="GID_0",
    title=f"{selected_var_name} ({docs_dict['units']})",
    markers=True,
    template="plotly_white",
)
var_year_px.update_xaxes(tickformat="%Y")
var_year_px.update_layout(
    legend_title="Country",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(var_year_px, use_container_width=True)

# Download button for national trends data
st.download_button(
    label="Download National Trends Data as CSV",
    data=convert_df(var_year),
    file_name=f"{selected_country}_national_trends_{selected_var_name}.csv",
    mime="text/csv",
)

# ----------------
# Plot the time series of the rank for the selected country for the selected variable
# Filter data
var_rank_year = (
    glocal_0_rank.loc[
        (glocal_0_rank["GID_0"].isin(selected_countries))
        & (glocal_0_rank.year.between(*selected_year)),
        ["year", "GID_0", selected_var],
    ]
    .dropna()
    .sort_values(["year", "GID_0"])
)
# Lineplot
var_rank_year_px = px.line(
    var_rank_year,
    x="year",
    y=selected_var,
    color="GID_0",
    title=f"Rank for variable {selected_var_name}",
    markers=True,
    labels={"x": "Year", "y": f"Rank for {selected_var_name}"},
    template="plotly_white",
)
var_rank_year_px.update_xaxes(tickformat="%Y")
var_rank_year_px.update_layout(
    legend_title="Country",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
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

    Subnational trends for {selected_var_name} averaged over the years: {selected_year[0]}-{selected_year[1]}, at GADM level {subnational_gadm_level} administrative boundaries.
    
    Note that administrative boundaries are obtained from [GADM v3.6](https://gadm.org/download_world36.html), and are slightly simplified for display purposes.
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

# Merge in gid names
gid_names = get_country_shapefile(subnational_gadm_level, selected_country)
glocal_subnational = glocal_subnational.merge(
    gid_names, on=f"GID_{subnational_gadm_level}", how="left"
)

# Download button for subnational trends data
st.download_button(
    label="Download Subnational Trends Data",
    data=convert_df(glocal_subnational),
    file_name=f"{selected_country}_subnational_trends_{selected_var_name}.csv",
    mime="text/csv",
)


# Read country shapefile
gdf_json = get_country_shapefile_as_geojson(subnational_gadm_level, selected_country)

# Plotly choropleth map
chropleth_px = px.choropleth(
    glocal_subnational,
    geojson=gdf_json,
    locations=f"GID_{subnational_gadm_level}",
    featureidkey=f"properties.GID_{subnational_gadm_level}",
    color=selected_var,
    color_continuous_scale="Viridis",
    hover_data=f"NAME_{subnational_gadm_level}",
    labels={
        selected_var: selected_var_name + " (" + docs_dict["units"] + ")",
        f"GID_{subnational_gadm_level}": "GID",
        f"NAME_{subnational_gadm_level}": "Region",
    },
)
if selected_country == "USA":
    chropleth_px.update_geos(scope="usa")
else:
    chropleth_px.update_geos(fitbounds="locations")

chropleth_px.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(chropleth_px, use_container_width=True)
