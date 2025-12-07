import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px

# 1. Taiwan Data
taiwan_data_csv = """Year,GDP
2000,14908
2001,13397
2002,13686
2003,14066
2004,15317
2005,16456
2006,16934
2007,17757
2008,18081
2009,16933
2010,19197
2011,20866
2012,21295
2013,21973
2014,22874
2015,22780
2016,23091
2017,25080
2018,25838
2019,25909
2020,28383
2021,33004
2022,32756
2023,32339
2024,34430
"""

# 2. World Bank Data (Placeholder - PASTE YOUR FULL DATA INSIDE THE QUOTES)
# Ensure it starts with """ and ends with """
raw_data_str = """Country Name	2000	2001	2002	2003	2004	2005	2006	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016	2017	2018	2019	2020	2021	2022	2023	2024
Czechia	6032.20	5753.13	5768.80	5982.68	6333.35	6631.84	6792.74	6983.36	7127.70	7450.85	7730.94	8057.21	8405.62	8923.84	9387.58	10036.52	10712.60	11177.01	10792.84	10998.67	11421.82	11544.55	11662.02	12076.20	12596.95
Korea, Rep.	12257.00	11561.00	12792.00	14039.00	15438.00	16870.00	18356.00	19876.00	18340.00	16632.00	20540.00	22489.00	22807.00	23842.00	25052.00	25082.00	25906.00	27608.00	29242.00	28675.00	28422.00	31296.00	29433.00	30058.00	31000.00
Singapore	23793.00	21577.00	22016.00	23573.00	27405.00	29870.00	33390.00	39224.00	39722.00	38578.00	46570.00	53094.00	54452.00	55618.00	56007.00	53630.00	53880.00	57714.00	62837.00	62422.00	58063.00	67855.00	72794.00	74609.00	76000.00
"""

# ==========================================
# 3. Data Processing
# ==========================================
def clean_currency(x):
    if isinstance(x, str):
        return pd.to_numeric(x.replace(',', '').strip(), errors='coerce')
    return x

@st.cache_data
def load_data():
    try:
        # Load Taiwan
        df_tw = pd.read_csv(io.StringIO(taiwan_data_csv))
        df_tw['Country'] = 'Taiwan'
        df_tw['GDP'] = pd.to_numeric(df_tw['GDP'], errors='coerce')
    except Exception:
        df_tw = pd.DataFrame()

    try:
        # Load World Bank
        if len(raw_data_str.strip()) > 50:
            df_wb = pd.read_csv(io.StringIO(raw_data_str), sep='\t', engine='python')
            if df_wb.shape[1] < 2:
                df_wb = pd.read_csv(io.StringIO(raw_data_str), sep=',', engine='python')

            col_map = {c: 'Country' for c in df_wb.columns if 'Country' in str(c)}
            df_wb = df_wb.rename(columns=col_map)
            year_cols = [c for c in df_wb.columns if str(c).strip().isdigit()]
            
            df_melted = df_wb.melt(id_vars=['Country'], value_vars=year_cols, var_name='Year', value_name='GDP')
            df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
            df_melted['GDP'] = df_melted['GDP'].apply(clean_currency)
            df_melted = df_melted.dropna(subset=['GDP', 'Year'])
            
            df_final = pd.concat([df_melted, df_tw], ignore_index=True)
        else:
            df_final = df_tw
            
    except Exception as e:
        st.error(f"Data Error: {e}")
        return df_tw

    if not df_final.empty:
        df_final['Year'] = df_final['Year'].astype(int)
        df_final = df_final.sort_values(['Country', 'Year'])
        
    return df_final

df = load_data()

# ==========================================
# 4. App Interface
# ==========================================
st.title("Analýza hospodářského růstu")

if df.empty:
    st.error("Žádná data k zobrazení.")
    st.stop()

all_countries = sorted(df['Country'].unique())
default_sel = ['Taiwan', 'Czechia']
selected_countries = st.sidebar.multiselect("Země", all_countries, default=[c for c in default_sel if c in all_countries])

min_y, max_y = int(df['Year'].min()), int(df['Year'].max())
year_range = st.sidebar.slider("Roky", min_y, max_y, (min_y, max_y))

df_filtered = df[(df['Country'].isin(selected_countries)) & (df['Year'].between(year_range[0], year_range[1]))]

tab1, tab2 = st.tabs(["Jedna země", "Konvergence"])

with tab1:
    if selected_countries:
        c = st.selectbox("Vyber zemi", selected_countries)
        dat = df_filtered[df_filtered['Country'] == c].sort_values('Year')
        
        if len(dat) > 1:
            X = dat['Year'].values
            y = dat['GDP'].values
            
            # Linear Regression
            X_reshaped = (X - X.min()).reshape(-1, 1)
            model = LinearRegression().fit(X_reshaped, y)
            y_pred = model.predict(X_reshaped)
            r2 = r2_score(y, y_pred)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Data'))
            fig.add_trace(go.Scatter(x=X, y=y_pred, mode='lines', name=f'Trend (R2={r2:.2f})'))
            st.plotly_chart(fig)
            st.write(f"Rovnice: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

with tab2:
    st.write("Beta Konvergence")
    beta_data = []
    for c in selected_countries:
        c_dat = df[df['Country'] == c]
        r_start = c_dat[c_dat['Year'] == year_range[0]]
        r_end = c_dat[c_dat['Year'] == year_range[1]]
        
        if not r_start.empty and not r_end.empty:
            y0 = r_start.iloc[0]['GDP']
            yT = r_end.iloc[0]['GDP']
            if y0 > 0:
                growth = (np.log(yT) - np.log(y0)) / (year_range[1] - year_range[0])
                beta_data.append({'Country': c, 'ln_Init': np.log(y0), 'Growth': growth})
    
    if len(beta_data) > 1:
        b_df = pd.DataFrame(beta_data)
        fig_b = px.scatter(b_df, x='ln_Init', y='Growth', text='Country', trendline='ols')
        st.plotly_chart(fig_b)
    else:
        st.warning("Nedostatek dat pro konvergenci.")