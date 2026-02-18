import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
import geopandas as gpd
from streamlit_folium import st_folium
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import geemap.foliumap as geemap
from geemap import basemaps
import ee
from RadGEEToolbox import GenericCollection, get_palette
from google.oauth2 import service_account
import json




# service_account = 'localpythonscripts@ut-gee-ugs-bsf-dev.iam.gserviceaccount.com'
# credentials = ee.ServiceAccountCredentials(service_account, 'C:\\Users\\mradwin\\ut-gee-ugs-bsf-dev-53dcc5d729e0.json')
# ee.Initialize(credentials=credentials)
try:
    # 1. Get the raw string from secrets
    key_content = st.secrets["textkey"]
    
    # 2. Parse JSON with 'strict=False'
    key_dict = json.loads(key_content, strict=False)
    
    # 3. Define the mandatory Earth Engine Scope
    #    This tells Google we want access to GEE specifically
    scopes = ['https://www.googleapis.com/auth/earthengine']
    
    # 4. Create Credentials WITH Scopes
    credentials = service_account.Credentials.from_service_account_info(
        key_dict, 
        scopes=scopes
    )
    
    # 5. Initialize
    ee.Initialize(credentials=credentials)
    
except Exception as e:
    # Fallback for Local Development
    local_key_path = 'C:\\Users\\mradwin\\ut-gee-ugs-bsf-dev-53dcc5d729e0.json'
    
    if os.path.exists(local_key_path):
        # The older helper function 'ee.ServiceAccountCredentials' automatically handles scopes
        # so we don't need to manually add them here.
        credentials = ee.ServiceAccountCredentials(
            'localpythonscripts@ut-gee-ugs-bsf-dev.iam.gserviceaccount.com', 
            local_key_path
        )
        ee.Initialize(credentials=credentials)
    else:
        st.error("🚨 Authentication Error")
        st.code(f"Detailed Error: {e}")
        st.stop()

def calculate_climatological_anomaly(
    collection, 
    baseline_start_year, 
    baseline_end_year, 
    frequency='yearly',
    band_name='Soil_Saturation_Percent_End_Of_Timestep',
    anomaly_band_name='Saturation_Anomaly_Climatological',
    debug=False
):
    """
    Calculates the anomaly relative to a baseline climatology.
    
    Args:
        frequency (str): 'monthly' (compares Jan to Jan avg) or 'yearly' (compares 2023 to 2000-2020 avg).
    """
    
    # 1. Establish the Baseline Collection
    # Filter strictly to the baseline period
    baseline_col = collection.filter(
        ee.Filter.calendarRange(baseline_start_year, baseline_end_year, 'year')
    ).select(band_name)
    
    # =========================================================
    # BRANCH A: MONTHLY CLIMATOLOGY (Seasonality Removal)
    # =========================================================
    if frequency == 'monthly':
        
        # 2A. Compute Climatology (12 images)
        months = ee.List.sequence(1, 12)
        
        def compute_month_mean(m):
            monthly_subset = baseline_col.filter(ee.Filter.calendarRange(m, m, 'month'))
            
            # Calculate mean
            mean_img = monthly_subset.mean().set('month', m)
            
            # Debug: Capture dates
            dates_used = monthly_subset.aggregate_array('Date_Filter') # or system:index
            return mean_img.set('clim_dates_used', dates_used)

        climatology = ee.ImageCollection.fromImages(months.map(compute_month_mean))
        
        # 3A. Join Climatology to Input
        # We need a 'month' property on the input to match against the climatology
        col_with_month = collection.map(lambda img: img.set('month', img.date().get('month')))
        
        filter_month = ee.Filter.equals(leftField='month', rightField='month')
        join = ee.Join.saveFirst(matchKey='clim_ref')
        
        # This creates a collection where each image has its specific monthly mean attached
        processing_col = ee.ImageCollection(join.apply(col_with_month, climatology, filter_month))

    # =========================================================
    # BRANCH B: YEARLY CLIMATOLOGY (Inter-annual Variability)
    # =========================================================
    elif frequency == 'yearly':
        
        # 2B. Compute Grand Mean (1 Single Image)
        # This is the average of ALL annual images in the baseline period
        global_mean_img = baseline_col.mean()
        
        # Debug: Capture all years used
        years_used = baseline_col.aggregate_array('year') # Assuming 'year' property exists
        
        # Add debug info to the single mean image so we can access it later
        global_mean_img = global_mean_img.set('clim_years_used', years_used)
        
        # 3B. Prepare Input for Broadcast
        # No join needed! We just attach this one image to every input image
        # so the mapping function below works identically for both branches.
        def attach_global_mean(img):
            return img.set('clim_ref', global_mean_img)
            
        processing_col = collection.map(attach_global_mean)
        
    else:
        raise ValueError("Frequency must be 'monthly' or 'yearly'")

    # =========================================================
    # 4. CALCULATE ANOMALY (Shared Logic)
    # =========================================================
    def compute_anomaly(img):
        # Retrieve the baseline image (either the specific Month or the Grand Year Mean)
        clim_img = ee.Image(img.get('clim_ref'))
        val = img.select(band_name)
        
        # Calculate Anomaly
        anom = val.subtract(clim_img).rename(anomaly_band_name)
        res = img.addBands(anom)
        
        if debug:
            if frequency == 'monthly':
                res = res.set('DEBUG_clim_dates_used', clim_img.get('clim_dates_used'))
                res = res.set('DEBUG_clim_month_used', clim_img.get('month'))
            elif frequency == 'yearly':
                res = res.set('DEBUG_clim_years_used', clim_img.get('clim_years_used'))
                
        return res
    
    return ee.ImageCollection(processing_col.map(compute_anomaly))

st.set_page_config(layout="wide", page_title="UBM App", page_icon="⚖️") # Use full screen width

st.image('Flaming_Gorge_Cropped_With_Logo.jpg', use_container_width=True)

st.markdown(
    """
    <style>
    /* Target the specific element Streamlit uses for Plotly */
    div[data-testid="stPlotlyChart"] {
        display: flex;
        justify-content: center;
    }
    /* Force st_folium to center itself */
    div[data-testid="stFolium"] {
        margin: auto;
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# with st.container(width='stretch', horizontal_alignment='center'):
#     st.logo('UGS-logo-large.png', size="large", link='https://geology.utah.gov/')

# --- SECTION 1: INFO SNIPPET ---
with st.container(horizontal_alignment='center'):
    # st.title("🌊 Utah Watersheds Explorer")
    # st.image('UGS-logo-large.png', width=100)
    st.header("⚖️ Utah Basin Model (UBM) Soil Water Balance Demo App (v1.1)", divider='rainbow', text_alignment='center')
    st.markdown("Developed by Mark Radwin and Paul Inkenbrandt at the Utah Geological Survey. Contact: mradwin@utah.gov GitHub: https://github.com/radwinskis/GEE_UBM", text_alignment='center')
    st.markdown("""
    #### **Welcome.** This app is designed to explore zonal statistics and maps of UBM ensemble runs across Utah. 
    
    ##### Select a region below to generate the ensemble time series, or scroll down to view spatial distribution maps.
    """, text_alignment='center')
    # st.header('', divider='rainbow', text_alignment='center')
    # st.divider()

# --- SECTION 2: REGION SELECTOR (Your "Navbar") ---
# Use session state to handle the "Map Mode" vs "Button Mode"
if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = 'Select from Map'



# --- 1. SETUP MAP ---
# Load your shapefile/GeoJSON of Utah Watersheds
@st.cache_data
def load_geodata():
    # 1. Read the files
    UT_basins_geojson = gpd.read_file("UT_HUC6_Basins.geojson")
    UT_watersheds_geojson = gpd.read_file("UT_Watersheds_Export.geojson")

    # --- FIX: UNPACK GEOMETRY COLLECTIONS ---
    def fix_geometry(geom):
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom
        if isinstance(geom, GeometryCollection):
            polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
            return unary_union(polys)
        return None

    def clean_gdf(gdf):
        gdf = gdf.copy()
        gdf['geometry'] = gdf['geometry'].apply(fix_geometry)
        gdf = gdf.dropna(subset=['geometry'])
        gdf = gdf[~gdf.geometry.is_empty]
        gdf['geometry'] = gdf.simplify(tolerance=0.001)
        if gdf.crs is None:
            gdf.set_crs(epsg=26912, inplace=True)
        if gdf.crs.to_string() != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)
        return gdf

    return clean_gdf(UT_basins_geojson), clean_gdf(UT_watersheds_geojson)

basins_gdf, watersheds_gdf = load_geodata()

# with st.container(width='stretch', horizontal_alignment='center'):
    # A horizontal radio acts like a navbar
boundary_choice = st.selectbox(
    "**Geometries to display on map:**",
    [ "UT Basins", "UT Watersheds"],
    index=0, width=400
)
selection_mode = st.radio(
    "**Choose Data Source for Zonal Statistics Timeseries Plot:**",
    ["UT Statewide", "Entire GSL Basin", "Select Geometry from Map"],
    horizontal=True,
    index=2
)
# Logic to handle the selection
# target_id = None
target_id = 'GSL_Basin_Watershed'  # Default selection

if st.session_state.get('boundary_choice') != boundary_choice:
    st.session_state['boundary_choice'] = boundary_choice
    st.session_state['last_map_clicked'] = None
    if st.session_state.get('selected_id') and selection_mode == "Select Watershed from Map":
        st.session_state['selected_id'] = None

active_gdf = watersheds_gdf if boundary_choice == "UT Watersheds" else basins_gdf
name_field = "HU_8_NAME" if boundary_choice == "UT Watersheds" else "Name"
layer_label = "Utah Watersheds" if boundary_choice == "UT Watersheds" else "Utah Basins"

m = folium.Map(location=[39.55, -111.5], zoom_start=7)

folium.GeoJson(
        active_gdf,
        name=layer_label,
        style_function=lambda x: {
            'fillColor': '#YlGn',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5,
        },
        highlight_function=lambda x: {
            'weight': 3,
            'color': 'black',
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=[name_field])
    ).add_to(m)

if selection_mode == "UT Statewide":
    target_id = "Utah_Statewide"
elif selection_mode == "Entire GSL Basin":
    target_id = "GSL_Basin_Watershed"
elif selection_mode == "Select Basin or Watershed from Map":

    # Create the base Folium map
    m = folium.Map(location=[39.55, -111.5], zoom_start=7)

    # m.addLayer(gdf)

    folium.GeoJson(
        active_gdf,
        name=layer_label,
        style_function=lambda x: {
            'fillColor': '#YlGn',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5,
        },
        highlight_function=lambda x: {
            'weight': 3,
            'color': 'black',
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(fields=[name_field])
    ).add_to(m)

map_output = st_folium(m, width=None, height=650, returned_objects=["last_active_drawing"])
if map_output["last_active_drawing"]:
    target_id = map_output["last_active_drawing"]["properties"].get(name_field)

# Fallback if nothing clicked yet
if not target_id and st.session_state.get('last_map_clicked'):
    target_id = st.session_state['last_map_clicked']

 # Save selection to state
if target_id:
    st.session_state['selected_id'] = target_id
    # If it came from the map, save it so it persists if we toggle modes
    if selection_mode == "Select Watershed from Map":
        st.session_state['last_map_clicked'] = target_id   


st.header("📈 Interactive Timeseries of UBM Ensemble Runs", divider='rainbow', text_alignment='center')
st.markdown("This figure is interactive. Hover over lines to see details, zoom in/out, and pan around.", text_alignment='center')

# --- PLOTTING SECTION ---
with st.container(horizontal=False, horizontal_alignment='center',  width='stretch'):

    if st.session_state.get('selected_id'):
    # Now we just look at the Session State, we don't care where it came from
        current_selection = st.session_state['selected_id']
        current_selection_filtered = current_selection.replace(',', '').replace("'", "").replace(" ", "_").replace("-", "_") .replace("__", "_") 
        
        # directory = 'C:\\Users\\mradwin\\Documents\\Utah Soil Water Balance\\Zonal_Stats_Timeseries\\All_Watersheds\\'
        base_directory = 'Zonal_Stats/UT_Basins/'
        if boundary_choice == "UT Watersheds":
            base_directory = 'Zonal_Stats/UT_Watersheds/'

        if current_selection:
            # st.subheader(f"Data for: {current_selection} AKA {current_selection_filtered}")
            
            directory = base_directory
            watershed = current_selection_filtered
            watershed_name = watershed.replace('_', ' ')
            folder_path = directory + watershed + '/'
            file_list = os.listdir(folder_path)
            # print(file_list)
            # master_df = pd.DataFrame()
            # master_df.columns = ['Date', 'Recharge_m3', 'Runoff_m3', 'Soil_Saturation_Percent_End_Of_Timestep', 'AET_m3', 'Precip_and_Snowmelt_m3', 'Irrigation_m3']
            recharge_df = pd.DataFrame()
            runoff_df = pd.DataFrame()
            soil_saturation_df = pd.DataFrame()
            AET_df = pd.DataFrame()
            precipitation_df = pd.DataFrame()
            irrigation_df = pd.DataFrame()

            def _find_col_by_substring(df, substring: str):
                matches = [c for c in df.columns if substring.lower() in c.lower()]
                return matches[0] if matches else None

            def _find_col_by_substrings(df, substrings):
                for substring in substrings:
                    match = _find_col_by_substring(df, substring)
                    if match:
                        return match
                return None

            def _extract_series(ws_df, date_col, value_col, new_name):
                if not date_col or not value_col:
                    return None
                subset = ws_df[[date_col, value_col]].copy()
                subset.rename(columns={date_col: "Date", value_col: new_name}, inplace=True)
                subset["Date"] = pd.to_datetime(subset["Date"], errors="coerce")
                return subset

            for file in file_list:
                if file.endswith('.csv'):
                    if 'DisALEXI' in file:
                        ET_type = 'OpenET_DisALEXI'
                    elif 'EEMETRIC' in file:
                        ET_type = 'OpenET_EEMETRIC'
                    elif 'PTJPL' in file:
                        ET_type = 'OpenET_PTJPL'
                    elif 'SSEBOP' in file:
                        ET_type = 'OpenET_SSEBOP'
                    elif 'GEESEBAL' in file:
                        ET_type = 'OpenET_GEESEBAL'
                    elif 'SIMS' in file:
                        ET_type = 'OpenET_SIMS'
                    else:
                        ET_type = 'Unknown_ET_Model'

                    if 'DAYMET' in file:
                        precip_type = 'DAYMET_Precipitation'
                    elif 'PRISM' in file:
                        precip_type = 'PRISM_Precipitation'
                    elif 'GRIDMET' in file:
                        precip_type = 'GRIDMET_Precipitation'
                    else:
                        precip_type = 'Unknown_Precipitation_Model'

                    if ET_type == 'OpenET_SIMS':
                        pass
                    else:
                    
                        file_path = os.path.join(folder_path, file)
                        ws_df = pd.read_csv(file_path)
                        date_col = _find_col_by_substring(ws_df, 'date')
                        recharge_col = _find_col_by_substring(ws_df, 'Recharge_m3')
                        runoff_col = _find_col_by_substring(ws_df, 'Runoff_m3')
                        soil_sat_col = _find_col_by_substring(ws_df, 'Soil_Saturation_Percent_End_Of_Timestep')
                        AET_col = _find_col_by_substring(ws_df, 'AET_m3')
                        precip_col = _find_col_by_substrings(
                            ws_df,
                            ['precip_and_snowmelt_input_m3', 'precip_and_snowmelt_input']
                        )
                        irrig_col = _find_col_by_substring(ws_df, 'irrigation_m3')

                        watershed_recharge_df = _extract_series(ws_df, date_col, recharge_col, f'Recharge_m3_{ET_type}_{precip_type}')
                        if watershed_recharge_df is not None:
                            recharge_df = pd.merge(recharge_df, watershed_recharge_df, on='Date', how='outer') if not recharge_df.empty else watershed_recharge_df

                        watershed_runoff_df = _extract_series(ws_df, date_col, runoff_col, f'Runoff_m3_{ET_type}_{precip_type}')
                        if watershed_runoff_df is not None:
                            runoff_df = pd.merge(runoff_df, watershed_runoff_df, on='Date', how='outer') if not runoff_df.empty else watershed_runoff_df

                        watershed_soil_sat_df = _extract_series(ws_df, date_col, soil_sat_col, f'Soil_Saturation_Percent_{ET_type}_{precip_type}')
                        if watershed_soil_sat_df is not None:
                            soil_saturation_df = pd.merge(soil_saturation_df, watershed_soil_sat_df, on='Date', how='outer') if not soil_saturation_df.empty else watershed_soil_sat_df

                        watershed_AET_df = _extract_series(ws_df, date_col, AET_col, f'AET_m3_{ET_type}_{precip_type}')
                        if watershed_AET_df is not None:
                            AET_df = pd.merge(AET_df, watershed_AET_df, on='Date', how='outer') if not AET_df.empty else watershed_AET_df

                        watershed_precipitation_df = _extract_series(ws_df, date_col, precip_col, f'Precip_and_Snowmelt_m3_{ET_type}_{precip_type}')
                        if watershed_precipitation_df is not None:
                            precipitation_df = pd.merge(precipitation_df, watershed_precipitation_df, on='Date', how='outer') if not precipitation_df.empty else watershed_precipitation_df

                        watershed_irrigation_df = _extract_series(ws_df, date_col, irrig_col, f'Irrigation_m3_{ET_type}_{precip_type}')
                        if watershed_irrigation_df is not None:
                            irrigation_df = pd.merge(irrigation_df, watershed_irrigation_df, on='Date', how='outer') if not irrigation_df.empty else watershed_irrigation_df
            M3_TO_ACFT = 0.000810714

            def _ensure_datetime_sorted(df: pd.DataFrame) -> pd.DataFrame:
                if df is None or df.empty:
                    return pd.DataFrame()
                out = df.copy()
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
                out = out.dropna(subset=["Date"]).sort_values("Date")
                return out

            def _numeric_cols(df: pd.DataFrame):
                cols = [c for c in df.columns if c != "Date"]
                # coerce to numeric (protects against stray strings)
                for c in cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                return cols

            def _trace_id_from_col(col_name: str) -> str:
                # As requested: split on "_" and take the last two tokens
                parts = str(col_name).split("_")
                return "_".join(parts[-4:]) if len(parts) >= 4 else str(col_name)

            def _select_one_per_precip_model(df: pd.DataFrame):
                """Keep ONE column containing PRISM, ONE containing GRIDMET, ONE containing DAYMET (first match in each)."""
                all_cols = [c for c in df.columns if c != "Date"]
                keep = []
                for key in ("PRISM", "GRIDMET", "DAYMET"):
                    matches = [c for c in all_cols if key.lower() in c.lower()]
                    if matches:
                        keep.append(matches[0])
                if not keep and all_cols:
                    keep.append(all_cols[0])
                return keep

            def _add_ensemble_subplot(fig, df, cols, row, title, y_scale, y_unit_label, hover_format):
                if df.empty or not cols:
                    return

                x = df["Date"]

                # Ensemble members (no legend; ID on hover)
                for c in cols:
                    tid = _trace_id_from_col(c)
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=df[c] * y_scale,
                            name=tid,
                            showlegend=False,
                            mode="lines",
                            line=dict(color="darkslategrey", width=1),
                            opacity=0.2,
                            hovertemplate=(
                                "%{x|%Y-%m-%d}<br>"
                                f"%{{y:{hover_format}}} {y_unit_label}<br>"
                                f"{tid}"
                                "<extra></extra>"
                            ),
                        ),
                        row=row,
                        col=1,
                    )

                # Ensemble mean (still no legend; labeled in hover)
                mean_series = df[cols].mean(axis=1, skipna=True)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean_series * y_scale,
                        name="Ensemble mean",
                        showlegend=False,
                        mode="lines",
                        line=dict(color="salmon", width=2.2),
                        opacity=0.8,
                        hovertemplate=(
                            "%{x|%Y-%m-%d}<br>"
                            f"%{{y:{hover_format}}} {y_unit_label}<br>"
                            "Ensemble mean<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=1,
                )

            # --- Prepare dataframes (assumes these already exist in your notebook) ---
            recharge_p = _ensure_datetime_sorted(recharge_df)
            runoff_p = _ensure_datetime_sorted(runoff_df)
            soil_saturation_p = _ensure_datetime_sorted(soil_saturation_df)
            AET_p = _ensure_datetime_sorted(AET_df)
            precip_p = _ensure_datetime_sorted(precipitation_df)
            irrig_p = _ensure_datetime_sorted(irrigation_df)

            recharge_cols = _numeric_cols(recharge_p) if not recharge_p.empty else []
            runoff_cols = _numeric_cols(runoff_p) if not runoff_p.empty else []
            soil_cols = _numeric_cols(soil_saturation_p) if not soil_saturation_p.empty else []
            AET_cols = _numeric_cols(AET_p) if not AET_p.empty else []

            # Precip: include all ensemble members (match other subplots)
            if not precip_p.empty:
                precip_cols = _numeric_cols(precip_p)
            else:
                precip_cols = []

            # Irrigation: only plot the first non-Date column
            if not irrig_p.empty:
                irrig_cols_all = _numeric_cols(irrig_p)
                irrig_col = irrig_cols_all[0] if irrig_cols_all else None
            else:
                irrig_col = None

            titles = (
                "Soil Saturation Percent (0-100)",
                "Recharge Volume",
                "Runoff Volume",
                "AET Volume",
                "Precipitation + Snowmelt Volume",
                "Irrigation Volume",
            )
            with st.container(width=1000):
                plot_scale = st.slider(
                    "Adjust slider to change overall size of figure",
                    min_value=0.7,
                    max_value=1.5,
                    value=1.0,
                    step=0.1,
                    help="Adjust plot size to fit your display."
                )

            base_height = 1200
            base_width = 1000
            scaled_height = int(base_height * plot_scale)
            scaled_width = int(base_width * plot_scale)

            fig = make_subplots(
                rows=6, cols=1,
                shared_xaxes=False,
                vertical_spacing=0.05,
                subplot_titles=titles,
                row_heights=[2, 2, 2, 2, 2, 1]
            )
            _add_ensemble_subplot(fig, soil_saturation_p, soil_cols, row=1, title=titles[0], y_scale=1.0, y_unit_label="%", hover_format=".1f")
            _add_ensemble_subplot(fig, recharge_p, recharge_cols, row=2, title=titles[1], y_scale=M3_TO_ACFT, y_unit_label="acre-ft", hover_format=",.0f")
            _add_ensemble_subplot(fig, runoff_p, runoff_cols, row=3, title=titles[2], y_scale=M3_TO_ACFT, y_unit_label="acre-ft", hover_format=",.0f")
            _add_ensemble_subplot(fig, AET_p, AET_cols, row=4, title=titles[3], y_scale=M3_TO_ACFT, y_unit_label="acre-ft", hover_format=",.0f")
            _add_ensemble_subplot(fig, precip_p, precip_cols, row=5, title=titles[4], y_scale=M3_TO_ACFT, y_unit_label="acre-ft", hover_format=",.0f")

            # Irrigation (single column only; no legend)
            if irrig_p is not None and not irrig_p.empty and irrig_col is not None:
                fig.add_trace(
                    go.Scatter(
                        x=irrig_p["Date"],
                        y=irrig_p[irrig_col] * M3_TO_ACFT,
                        name="Irrigation",
                        showlegend=False,
                        mode="lines",
                        line=dict(color="darkslategrey", width=1.6),
                        opacity=1.0,
                        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} acre-ft<br>Irrigation<extra></extra>",
                    ),
                    row=6, col=1
                )
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="salmon", width=2.2),
                    name="Ensemble mean",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color="darkslategrey", width=1.2),
                    opacity=0.8,
                    name="Ensemble runs",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

            # --- Styling (no legends) ---
            target_font = "Times New Roman"
            fig.update_layout(
                title=dict(
                    text=f"{watershed_name} — UBM Ensemble Time Series",
                    x=0.5,          # center
                    xanchor="center",
                    y=0.98,
                    yanchor="top",
                    font=dict(family=target_font, size=18, color="black"),
                ),
                height=scaled_height,
                width=scaled_width,
                template="plotly_white",      # optional, but helps ensure white defaults
                paper_bgcolor="white",        # <-- this is the outer background
                plot_bgcolor="white", 
                font=dict(family=target_font, size=14, color="black"),
                margin=dict(t=60, b=50, l=70, r=60),
                showlegend=True,
                legend=dict(
                    x=0.02, y=0.9,              # inside top subplot, near bottom-left-ish (tweak y if needed)
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.0)",
                    borderwidth=0,
                    font=dict(family=target_font, size=12, color="black"),
                    orientation="h",
                )
            )

            fig.update_annotations(font=dict(family=target_font, size=15))

            fig.update_xaxes(
                showline=True, linewidth=1.2, linecolor="black", mirror=True,
                showgrid=False,
                ticks="outside", ticklen=6,
                minor=dict(ticklen=4, dtick="M6", showgrid=False),
            )

            fig.update_yaxes(
                showline=True, linewidth=1.2, linecolor="black", mirror=True,
                showgrid=False,
                ticks="outside", ticklen=6,
                title_text="Volume (acre-ft)",
            )
            fig.update_yaxes(title_text="Soil Saturation (%)", row=1, col=1)


            fig.update_xaxes(title_text="Date", row=6, col=1, dtick="M12")
            for r in range(1, 7):
                fig.update_xaxes(
                    showticklabels=True,
                    row=r, col=1,
                    range=[pd.Timestamp('2005-01-01'), pd.Timestamp('2024-12-31')],
                    dtick="M12"
                )

            st.plotly_chart(fig, width=scaled_width, theme=None)
        else:
            st.info("👈 Select a watershed from the map OR click a button above.")

st.divider()

### Mapping Section ###
def convert_depth_to_volume(image, proj):
    """Converts pixel values from depth (mm) to volume (m^3)."""
    image = image.setDefaultProjection(proj)
    pixel_area = ee.Image.pixelArea().reproject(proj)
    depth_in_meters = image.multiply(0.001)
    volume_m3 = pixel_area.multiply(depth_in_meters)
    return volume_m3 #.copyProperties(image, image.propertyNames())

def _build_soil_saturation_anomaly(full_collection, year_value, native_proj):
    yearly_collection = GenericCollection(
        collection=full_collection.select('Soil_Saturation_Percent_End_Of_Timestep')
    ).yearly_mean_collection()
    anomaly_collection = calculate_climatological_anomaly(
        collection=yearly_collection.collection,
        baseline_start_year=2005,
        baseline_end_year=2024
    ).filter(ee.Filter.eq('year', ee.Number(year_value)))
    return anomaly_collection.select('Saturation_Anomaly_Climatological').mean().setDefaultProjection(native_proj)
with st.container(width='stretch', horizontal_alignment='center'):
    st.header("🗺️ Spatial Distribution Analysis", divider='rainbow', text_alignment='center')
    st.markdown("""
            ##### Explore spatial distributions of Soil Water Balance Model outputs and inputs across Utah for selected years and variables.
                
            Select the year, variable, unit, and model from the dropdowns below to generate the maps. Adjust the min and max sliders to adjust color scaling as needed.      
            """, text_alignment='center')
    with st.container(width=1500):
        with st.expander("More info"):
            st.markdown("""
                  

            ##### Explanation of Map Layer Variables:
            - Soil Saturation Percent Anomaly: The deviation of soil saturation percent from the mean of the timeseries. For yearly mean products (app default), this represents the yearly mean deviation from the mean of the timeseries. This anomaly product highlights drier-than-normal or wetter-than-normal regions and periods.
            - Soil Saturation Percent: The percent of soil saturation for the shallow portion of soil modelled for each pixel. This represents the upper-most soil column and is not reflective of the total soil column.   
            - Recharge: The amount of water that drains out of the modelled soil column, representing effective recharge into the subsurface.
            - Runoff: The amount of water that spills over the soil modelled soil column, representing water that could not be incorporated as soil water or recharge.
            - AET: The amount of water evaporated from the soil or transpired from vegetation. The primary 'out' variable of the model, provided by OpenET.
            - Precipitation + Snowmelt: The amount of water inputs from precipitation as rain or from snowmelt.
            - Irrigation: The amount of human introduced water for agricultural purposes, derived from the 2024 Utah Water Budget dataset and Water Related Land Use polygons.   
            
            > NOTE: DAYMET derived models have a finer resolution (1km) compared to PRISM and GRIDMET derived models (4.5km). Thus, volumetric values for DAYMET are smaller as there is less volume of water per pixel.
            """, text_alignment='center')
  
    with st.container(width=1500, horizontal_alignment='center'):
        col_controls1, col_controls2, col_controls3, col_controls4 = st.columns([2, 2, 2, 1]) #st.columns(4)
        with col_controls1:
            year_select = st.slider("Select Year", 2005, 2024, 2024)
        with col_controls2:
            variable_select = st.selectbox("Select Variable", ["Soil Saturation Percent Anomaly", "Soil Saturation Percent", "Recharge", "Runoff", "AET", "Precipitation + Snowmelt", "Irrigation"])
        with col_controls3:

            model_select = st.selectbox("Select Model", ["Ensemble Mean", "OpenET DisALEXI & DAYMET Precipitation", "OpenET EEMETRIC & DAYMET Precipitation", 
                                                        "OpenET PTJPL & DAYMET Precipitation", "OpenET SSEBOP & DAYMET Precipitation", "OpenET GEESEBAL & DAYMET Precipitation", 
                                                        "OpenET DisALEXI & PRISM Precipitation", "OpenET EEMETRIC & PRISM Precipitation", 
                                                        "OpenET PTJPL & PRISM Precipitation", "OpenET SSEBOP & PRISM Precipitation", "OpenET GEESEBAL & PRISM Precipitation", 
                                                        "OpenET DisALEXI & GRIDMET Precipitation", "OpenET EEMETRIC & GRIDMET Precipitation", 
                                                        "OpenET PTJPL & GRIDMET Precipitation", "OpenET SSEBOP & GRIDMET Precipitation", "OpenET GEESEBAL & GRIDMET Precipitation", 
                                                        ])
        with col_controls4:
            if variable_select in ("Soil Saturation Percent", "Soil Saturation Percent Anomaly"):
                unit = st.selectbox("Select Unit", ["Percentage %"], index=0)
            else:
                unit = st.selectbox("Select Unit", ["Acre-Feet (acre-ft)", "Cubic Meters (m³)", "Depth (mm)"], index=0)
    bands_dict = {"Soil Saturation Percent Anomaly": "Saturation_Anomaly_Climatological", "Soil Saturation Percent": "Soil_Saturation_Percent_End_Of_Timestep", "Recharge": "Recharge", "Runoff": "Runoff", "AET":"AET", 
                  "Precipitation + Snowmelt":"precip_and_snowmelt_input", "Irrigation":"irrigation"}
    
    if variable_select == "Soil Saturation Percent Anomaly":
        unit_scalar = 1.0
        unit_label = "%"
    elif variable_select == "Soil Saturation Percent":
        unit_scalar = 1.0
        unit_label = "%"
    elif unit == "Percentage %":
        unit_scalar = 1.0
        unit_label = "%"
    elif unit == "Acre-Feet (acre-ft)":
        unit_scalar = M3_TO_ACFT
        unit_label = "acre-ft"
    elif unit == "Cubic Meters (m³)":
        unit_scalar = 1.0
        unit_label = "m³"
    elif unit == "Depth (mm)":
        unit_scalar = 1.0
        unit_label = "mm"

    convert_to_volume = unit != "Depth (mm)"
    depth_vars = ("AET", "Precipitation + Snowmelt", "Irrigation", "Recharge", "Runoff")
    is_depth_var = variable_select in depth_vars
    
    if model_select == "Ensemble Mean":
        model_key = "Ensemble_Mean"
        col1 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETDALEXI_IRRIm_M_mm')
        col2 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETEMTRIC_IRRIm_M_mm')
        col3 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETGSEBAL_IRRIm_M_mm')
        col4 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETPTJPL_IRRIm_M_mm')
        col5 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETSBOP_IRRIm_M_mm')
        col6 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETSIMS_IRRIm_M_mm')
        col7 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETDALEXI_IRRIm_M_mm')
        col8 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETEMTRIC_IRRIm_M_mm')
        col9 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETGSEBAL_IRRIm_M_mm')
        col10 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETPTJPL_IRRIm_M_mm')
        col11 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETSBOP_IRRIm_M_mm')
        col12 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETSIMS_IRRIm_M_mm')
        col13 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETDALEXI_IRRIm_M_mm')
        col14 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETEMTRIC_IRRIm_M_mm')
        col15 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETGSEBAL_IRRIm_M_mm')
        col16 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETPTJPL_IRRIm_M_mm')
        col17 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETSBOP_IRRIm_M_mm')
        col18 = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETSIMS_IRRIm_M_mm')
        def reduceResolution(img, work_proj=None, reducer_type='sum'):
            if reducer_type == 'sum':
                reducer = ee.Reducer.sum()
            elif reducer_type == 'mean':
                reducer = ee.Reducer.mean()
            target_proj = col13.first().projection() #ee.Projection('EPSG:32612').atScale(1000)
            # work_proj: fine-scale metric grid to run the kernel on
            wp = work_proj or img.projection()
            # img_fine = img.reproject(wp)
            img_fine = img.setDefaultProjection(wp)
            agg = img_fine.reduceResolution(reducer=reducer, maxPixels=65536)
            return agg.reproject(target_proj) #.set('system:time_start', img.get('system:time_start'))
        # cols = [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16, col17, col18]
        cols = [col1, col2, col3, col4, col5, col7, col8, col9, col10, col11, col13, col14, col15, col16, col17]
        stacked_collection = []
        if variable_select == 'Soil Saturation Percent Anomaly':
            for i, col in enumerate(cols):
                yearly_collection = GenericCollection(collection=col.select('Soil_Saturation_Percent_End_Of_Timestep')).yearly_mean_collection()
                yearly_climatological_anomaly_col = calculate_climatological_anomaly(collection=yearly_collection.collection,
                                                                                     baseline_start_year=2005, baseline_end_year=2024)
                filtered_col = yearly_climatological_anomaly_col.filter(ee.Filter.eq('year', ee.Number(year_select))).select('Saturation_Anomaly_Climatological').first()
                if i < 5:
                    filtered_col = reduceResolution(filtered_col, col1.first().projection(), reducer_type='mean')
                stacked_collection.append(filtered_col)
        else:
            for i, col in enumerate(cols):
                col = col.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
                col = col.select(bands_dict[variable_select])
                native_proj = col.first().projection()
                if variable_select == "Soil Saturation Percent":
                    col = col.mean()
                else:
                    col = col.sum()
                if is_depth_var and convert_to_volume:
                    col = convert_depth_to_volume(ee.Image(col), native_proj)
                if i < 5:
                    if variable_select == "Soil Saturation Percent":
                        col = reduceResolution(col, col1.first().projection(), reducer_type='mean')
                    else:
                        if convert_to_volume is True:
                            col = reduceResolution(col, col1.first().projection())
                        else:
                            col = reduceResolution(col, col1.first().projection(), reducer_type='mean')
                
                stacked_collection.append(col)     
        stacked_collection = ee.ImageCollection(stacked_collection)
        image = stacked_collection.mean()

    elif model_select == "OpenET DisALEXI & DAYMET Precipitation":
        model_key = "ETDisALEXI_DAYMETSNOM"
        collection = ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETDALEXI_IRRIm_M_mm').filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        if variable_select == "Soil Saturation Percent":
            native_proj = collection.first().select(bands_dict[variable_select]).projection()
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            collection = GenericCollection(collection=ee.ImageCollection('projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETDALEXI_IRRIm_M_mm')).yearly_mean_collection()
            anomaly_collection = calculate_climatological_anomaly(collection=collection.collection.select('Soil_Saturation_Percent_End_Of_Timestep'),
                                                      baseline_start_year=2005, baseline_end_year=2024).filter(ee.Filter.eq('year', ee.Number(year_select)))
            image = anomaly_collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        else:
            native_proj = collection.first().select(bands_dict[variable_select]).projection()
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET EEMETRIC & DAYMET Precipitation":
        model_key = "ETEEMETRIC_DAYMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETEMTRIC_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET PTJPL & DAYMET Precipitation":
        model_key = "ETPTJPL_DAYMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETPTJPL_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET SSEBOP & DAYMET Precipitation":
        model_key = "ETSSEBOP_DAYMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETSBOP_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET GEESEBAL & DAYMET Precipitation":
        model_key = "ETGEESEBAL_DAYMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETGSEBAL_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET SIMS & DAYMET Precipitation":
        model_key = "ETSIMS_DAYMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_DAYMETSNOM_ETSIMS_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET DisALEXI & PRISM Precipitation":
        model_key = "ETDisALEXI_PRISMSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETDALEXI_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET EEMETRIC & PRISM Precipitation":
        model_key = "ETEEMETRIC_PRISMSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETEMTRIC_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET PTJPL & PRISM Precipitation":
        model_key = "ETPTJPL_PRISMSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETPTJPL_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET SSEBOP & PRISM Precipitation":
        model_key = "ETSSEBOP_PRISMSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETSBOP_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET GEESEBAL & PRISM Precipitation":
        model_key = "ETGEESEBAL_PRISMSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETGSEBAL_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET SIMS & PRISM Precipitation":
        model_key = "ETSIMS_PRISMSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_PRISMSNOM_ETSIMS_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET DisALEXI & GRIDMET Precipitation":
        model_key = "ETDisALEXI_GRIDMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETDALEXI_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET EEMETRIC & GRIDMET Precipitation":
        model_key = "ETEEMETRIC_GRIDMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETEMTRIC_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj) 
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET PTJPL & GRIDMET Precipitation":
        model_key = "ETPTJPL_GRIDMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETPTJPL_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET SSEBOP & GRIDMET Precipitation":
        model_key = "ETSSEBOP_GRIDMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETSBOP_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET GEESEBAL & GRIDMET Precipitation":
        model_key = "ETGEESEBAL_GRIDMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETGSEBAL_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
    elif model_select == "OpenET SIMS & GRIDMET Precipitation":
        model_key = "ETSIMS_GRIDMETSNOM"
        asset_id = 'projects/ut-gee-ugs-bsf-dev/assets/ModifiedUBM1Runs/Mod_UBM_1_RF1kmST_POLPor_OLMFC_HHSWP_NGMDGKSdM_GRIDMETSNOM_ETSIMS_IRRIm_M_mm'
        full_collection = ee.ImageCollection(asset_id)
        collection = full_collection.filterDate(f'{year_select}-01-01', f'{year_select}-12-31')
        native_proj = collection.first().select(bands_dict[variable_select]).projection()
        if variable_select == "Soil Saturation Percent":
            image = collection.select(bands_dict[variable_select]).mean().setDefaultProjection(native_proj)
        elif variable_select == 'Soil Saturation Percent Anomaly':
            native_proj = full_collection.first().select(['Soil_Saturation_Percent_End_Of_Timestep']).projection()
            image = _build_soil_saturation_anomaly(full_collection, year_select, native_proj)
        else:
            if is_depth_var and convert_to_volume:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)
                image = convert_depth_to_volume(ee.Image(image), native_proj)
            else:
                image = collection.select(bands_dict[variable_select]).sum().setDefaultProjection(native_proj)



    # if variable_select in ["AET", "Precipitation + Snowmelt", "Irrigation"]:
    #     image = convert_depth_to_volume(ee.Image(image))
    # else:
    #     pass
      
    def _int_slider(label, min_value, max_value, value):
        if max_value <= min_value:
            max_value = min_value + 1
        step = max(1, int((max_value - min_value) / 100))
        return st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step)

    with st.container(width=1000, horizontal_alignment='center'):
        viz_controls1, viz_controls2 = st.columns(2)
        with viz_controls1:
            if variable_select == "Soil Saturation Percent Anomaly":
                min_select = _int_slider("Select Min Stretch Value", min_value=-100, max_value=0, value=-15)
            if variable_select == "Soil Saturation Percent":
                min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=100, value=0)
            if variable_select == "Recharge":
                if unit == "Depth (mm)":
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=1000, value=0)
                else:
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=int(1E6*unit_scalar), value=int(0*unit_scalar))
            if variable_select == "Runoff":
                if unit == "Depth (mm)":
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=500, value=0)
                else:
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=int(1E5*unit_scalar), value=int(0*unit_scalar))
            if variable_select == "AET":
                if unit == "Depth (mm)":
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=1500, value=100)
                elif 'DAYMET' in model_select:
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=int(1E5*unit_scalar), value=int(1E4*unit_scalar))
                else:
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=int(5E6*unit_scalar), value=int(1E6*unit_scalar))
            if variable_select == "Precipitation + Snowmelt":
                if unit == "Depth (mm)":
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=2000, value=0)
                else:
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=int(1E6*unit_scalar), value=int(0*unit_scalar))
            if variable_select == "Irrigation":
                if unit == "Depth (mm)":
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=1500, value=0)
                elif 'DAYMET' in model_select:
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=int(1E5*unit_scalar), value=int(0*unit_scalar))
                else:
                    min_select = _int_slider("Select Min Stretch Value", min_value=0, max_value=int(1E6*unit_scalar), value=int(0*unit_scalar))
        with viz_controls2:
            if variable_select == "Soil Saturation Percent Anomaly":
                max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=100, value=15)
            if variable_select == "Soil Saturation Percent":
                max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=100, value=100)
            if variable_select == "Recharge":
                if unit == "Depth (mm)":
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=2500, value=1000)
                elif 'DAYMET' in model_select:
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=int(3E6*unit_scalar), value=int(1E6*unit_scalar))
                else:
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=int(3E7*unit_scalar), value=int(1.5E7*unit_scalar))
            if variable_select == "Runoff":
                if unit == "Depth (mm)":
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=1500, value=500)
                elif 'DAYMET' in model_select:
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=int(1.5E6*unit_scalar), value=int(7E5*unit_scalar))
                else:
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=int(2E7*unit_scalar), value=int(8E6*unit_scalar))
            if variable_select == "AET":
                if unit == "Depth (mm)":
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=2000, value=1200)
                elif 'DAYMET' in model_select:
                    max_select = _int_slider("Select Max Stretch Value", min_value=0, max_value=int(5E6*unit_scalar), value=int(2E6*unit_scalar))
                else:
                    max_select = _int_slider("Select Max Stretch Value", min_value=int(1E7*unit_scalar), max_value=int(3.5E7*unit_scalar), value=int(3E7*unit_scalar))
            if variable_select == "Precipitation + Snowmelt":
                if unit == "Depth (mm)":
                    max_select = _int_slider("Select Max Stretch Value", min_value=500, max_value=3000, value=1500)
                elif 'DAYMET' in model_select:
                    max_select = _int_slider("Select Max Stretch Value", min_value=int(0.7E6*unit_scalar), max_value=int(2.5E6*unit_scalar), value=int(1.5E6*unit_scalar))
                else:
                    max_select = _int_slider("Select Max Stretch Value", min_value=int(1E7*unit_scalar), max_value=int(5E7*unit_scalar), value=int(3E7*unit_scalar))
            if variable_select == "Irrigation":
                if unit == "Depth (mm)":
                    max_select = _int_slider("Select Max Stretch Value", min_value=500, max_value=2000, value=1000)
                elif 'DAYMET' in model_select:
                    max_select = _int_slider("Select Max Stretch Value", min_value=int(5E5*unit_scalar), max_value=int(2E6*unit_scalar), value=int(1E6*unit_scalar))
                else:
                    max_select = _int_slider("Select Max Stretch Value", min_value=int(0.25E7*unit_scalar), max_value=int(3E7*unit_scalar), value=int(1.5E7*unit_scalar))
            
    # --- SECTION 5: THE SPATIAL MAP (GEE) ---
    # Use geemap for GEE integration
    with st.container(width='stretch', horizontal_alignment='center'):
        try:
            # 1. Initialize State for the GEE Map
            if 'gee_last_click' not in st.session_state:
                st.session_state['gee_last_click'] = None
            # Initialize the map (centered on Utah)
            # Note: ipyleaflet is supported via geemap, but geemap.foliumap is often more stable in Streamlit
            Map = geemap.Map(center=[39.5, -111.5], zoom=7)
            Map.add_basemap('Esri.WorldShadedRelief')

            # 1. Add the Layer (Your existing logic)
            layer_name = f'{variable_select} {model_key} {year_select}'
            
            if variable_select == "Soil Saturation Percent Anomaly":
                Map.addLayer(image, {'min': min_select, 'max': max_select, 'palette': get_palette('rdbu')}, f'{variable_select} {model_key} {year_select}')
                palette = get_palette('rdbu')
            elif variable_select == "Soil Saturation Percent":
                Map.addLayer(image, {'min': min_select, 'max': max_select, 'palette': get_palette('rdylbu')}, f'{variable_select} {model_key} {year_select}')
                palette = get_palette('rdylbu')
                # Map.add_colorbar({'min': min_select, 'max': max_select, 'palette': get_palette('rdylbu')}, label=f'Soil Water Volume ({unit_label})', position='bottomright', background_color='white')
            elif variable_select == "Recharge":
                Map.addLayer(image.multiply(unit_scalar), {'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, f'{variable_select} {model_key} {year_select}')
                tick_labels = [f"{min_select:,.0f}", f"{max_select:,.0f}"]
                palette = get_palette('blues')
                # Map.add_colorbar({'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, label=f'Recharge Volume ({unit_label})', position='bottomright', background_color='white', tick_labels=tick_labels)
            elif variable_select == "Runoff":
                Map.addLayer(image.multiply(unit_scalar), {'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, f'{variable_select} {model_key} {year_select}')
                palette = get_palette('blues')
                # Map.add_colorbar({'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, label=f'Runoff Volume ({unit_label})', position='bottomright', background_color='white')
            elif variable_select == "AET":
                Map.addLayer(image.multiply(unit_scalar), {'min': min_select, 'max': max_select, 'palette': get_palette('evapotranspiration')}, f'{variable_select} {model_key} {year_select}')
                palette = get_palette('evapotranspiration')
                # Map.add_colorbar({'min': min_select, 'max': max_select, 'palette': get_palette('evapotranspiration')}, label=f'AET Volume ({unit_label})', position='bottomright', background_color='white')
            elif variable_select == "Precipitation + Snowmelt":
                Map.addLayer(image.multiply(unit_scalar), {'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, f'{variable_select} {model_key} {year_select}')
                palette = get_palette('blues')
                # Map.add_colorbar({'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, label=f'Precipitation + Snowmelt Volume ({unit_label})', position='bottomright', background_color='white')
            elif variable_select == "Irrigation":
                Map.addLayer(image.multiply(unit_scalar), {'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, f'{variable_select} {model_key} {year_select}')
                palette = get_palette('blues')
                # Map.add_colorbar({'min': min_select, 'max': max_select, 'palette': get_palette('blues')}, label=f'Irrigation Volume ({unit_label})', position='bottomright', background_color='white')
            # Map.add_basemap('ROADMAP')
            label_url = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png"

        # ... (After your Map.addLayer calls) ...

            # --- CLEAN LEGEND LOGIC ---
            # 1. Determine Scale Factor (e.g., divide by 1 Million)
            #    This keeps the labels short (e.g., "5.0" instead of "5,000,000")
            if max_select >= 1_000_000:
                scale_factor = 1_000_000
                legend_unit_label = f"(1E6 {unit_label})"
            elif max_select >= 1_000:
                scale_factor = 1_000
                legend_unit_label = f"(1E3 {unit_label})"
            else:
                scale_factor = 1
                legend_unit_label = f"({unit_label})"

            # 2. Create Scaled Min/Max for the Legend ONLY
            legend_min = min_select / scale_factor
            legend_max = max_select / scale_factor

            # 3. Add the Colorbar with Scaled Values
            #    Note: We pass a new 'viz_params' dict just for this legend.
            #    The 'palette' ensures the colors match the map perfectly.
            legend_descriptor = "Volume"
            if variable_select == "Soil Saturation Percent":
                legend_descriptor = "Percent"
            elif unit == "Depth (mm)":
                legend_descriptor = "Depth"

            Map.add_colorbar(
                vis_params={
                    'min': legend_min,
                    'max': legend_max,
                    'palette': palette
                },
                label=f'Per-Pixel {variable_select} {legend_descriptor} {legend_unit_label}',
                ticks=[legend_min, legend_max]
            ) 
        
            folium.TileLayer(
                tiles=label_url,
                attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                name='Streets & Labels',
                overlay=True,  # <--- CRITICAL: Tells the map this is see-through
                opacity=1.0,
                control=True
            ).add_to(Map)
            # Display the map
            # Map.to_streamlit(height=800, width=1300)
            # 2. Check for Previous Click in Session State (to persist marker across re-runs)
            if st.session_state['gee_last_click']:
                lat = st.session_state['gee_last_click']['lat']
                lng = st.session_state['gee_last_click']['lng']
                # folium.Marker([lat, lng], tooltip="Selected Pixel").add_to(Map)

            # 3. Render the Map with st_folium to capture interaction
            #    We replace Map.to_streamlit() with st_folium() directly
            st.subheader("Interactive Map (Click to Query Pixel Value)", divider='gray', text_alignment='center')

            map_key = f"{model_select}_{variable_select}_{year_select}"
        # Centering Hack: 3 columns. 
            # [1, 10, 1] means small spacers on sides, wide content in middle.
            c1, c2, c3 = st.columns([1, 10, 1])
            with c2:
                # We capture the output here
                map_output = st_folium(Map, height=700, width=1300, returned_objects=["last_clicked"], key=map_key)
            
            # with st.container(width='stretch', horizontal_alignment='center'):
            # map_output = st_folium(Map, height=700, width=800, returned_objects=["last_clicked"], key=map_key)

            # 5. Handle New Clicks (The State Updater)
            if map_output['last_clicked']:
                # If the click is different from what we have stored...
                if st.session_state['gee_last_click'] != map_output['last_clicked']:
                    st.session_state['gee_last_click'] = map_output['last_clicked']
                    # st.rerun() # RELOAD PAGE to draw the marker

            # 6. Display Data (The Result Viewer)
            # This runs on the RELOADED page because it looks at session_state, not map_output
            if st.session_state['gee_last_click']:
                
                # Get coords from state
                click_lat = st.session_state['gee_last_click']['lat']
                click_lng = st.session_state['gee_last_click']['lng']
                
                # Visual separator
                st.divider()
                st.markdown(f"### 🌎 Pixel Analysis: {click_lat:.4f}, {click_lng:.4f}", text_alignment='center')
                
                # Query Earth Engine
                # Use a spinner so the user knows something is happening
                with st.spinner("Querying Earth Engine..."):
                    point = ee.Geometry.Point([click_lng, click_lat])
                    
                    # Sample the image at 1km scale
                    sample = image.reduceRegion(
                        reducer=ee.Reducer.first(), 
                        geometry=point, 
                        scale=1000,  # Match your data resolution
                        bestEffort=True
                    ).getInfo()
                    
                # Display Result

                if sample:
                    val = list(sample.values())[0]
                    if val is None:
                        st.warning("Selected pixel is masked (No Data).")
                    else:
                        display_val = val
                        if variable_select != "Soil Saturation Percent":
                            display_val = val * unit_scalar
                        # Use a big metric to show the value clearly
                        # st.metric(
                        #     label=f"{variable_select} ({year_select})", 
                        #     value=f"{val:,.0f} {unit_label}"
                        # )
                        st.markdown(f"### Value at pixel for {variable_select} of {year_select} ({model_select} member):", text_alignment='center')
                        # st.markdown(f":large[{val:,.0f} {unit_label}]", text_alignment='center')
                        st.subheader(f'''{display_val:,.0f} {unit_label} ''', text_alignment='center')
                else:
                    st.warning("Could not retrieve value.")
            
        except Exception as e:
            st.warning("Earth Engine not initialized or geemap not installed.")
            st.error(e)
