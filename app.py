import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

# -----------------------------------------------------------------------------
# 1. 內建台灣數據 (Hardcoded User Data - Taiwan)
# -----------------------------------------------------------------------------
# 這是您提供的台灣專屬數據，無論如何都會存在
taiwan_data_csv = """Year,GDP
1971,2068.34
1972,2307.78
1973,2556.12
1974,2577.74
1975,2687.76
1976,3010.52
1977,3287.91
1978,3665.41
1979,3913.33
1980,4148.71
1981,4363.10
1982,4492.07
1983,4819.11
1984,5225.90
1985,5402.50
1986,5956.21
1987,6645.33
1988,7098.00
1989,7634.08
1990,7968.33
1991,8540.84
1992,9160.99
1993,9694.07
1994,10328.80
1995,10906.44
1996,11486.23
1997,12072.82
1998,12464.32
1999,13197.77
2000,13921.04
2001,13629.68
2002,14298.97
2003,14837.32
2004,15809.82
2005,16600.12
2006,17485.93
2007,18607.11
2008,18690.28
2009,18324.38
2010,20147.16
2011,20839.84
2012,21232.32
2013,21690.29
2014,22656.57
2015,22930.45
2016,23374.06
2017,24189.24
2018,24866.13
2019,25608.55
2020,26499.99
2021,28418.13
2022,29366.31
2023,29656.23
2024,31127.85"""

# -----------------------------------------------------------------------------
# 2. 數據處理核心 (Data Processing)
# -----------------------------------------------------------------------------
def clean_currency(x):
    """清理數據中的逗號與非數值字符"""
    if isinstance(x, str):
        # 移除逗號
        clean_str = x.replace(',', '').strip()
        try:
            return float(clean_str)
        except ValueError:
            return np.nan
    return x

def process_data(uploaded_file=None, pasted_text=None):
    df_wb = pd.DataFrame()
    
    # 1. 讀取 World Bank 數據 (優先權：上傳檔案 > 貼上文字)
    try:
        if uploaded_file is not None:
            # 判斷檔案類型
            if uploaded_file.name.endswith('.csv'):
                # 嘗試跳過 World Bank 檔案常見的前 4 行 Metadata
                try:
                    df_wb = pd.read_csv(uploaded_file, skiprows=4)
                    if 'Country Name' not in df_wb.columns: # 如果沒有 skiprows 4 讀不到，試試直接讀
                        uploaded_file.seek(0)
                        df_wb = pd.read_csv(uploaded_file)
                except:
                    uploaded_file.seek(0)
                    df_wb = pd.read_csv(uploaded_file)
            else:
                df_wb = pd.read_excel(uploaded_file)
                
        elif pasted_text and pasted_text.strip():
            # 處理貼上的文字 (通常是 Tab 分隔)
            try:
                df_wb = pd.read_csv(io.StringIO(pasted_text), sep='\t')
            except:
                # 如果 tab 失敗，嘗試用逗號
                df_wb = pd.read_csv(io.StringIO(pasted_text), sep=',')

        # 2. 清洗 World Bank 格式
        if not df_wb.empty:
            # 確保有 Country Name 欄位 (有些檔案叫 'Country', 有些叫 'Country Name')
            col_map = {c: c for c in df_wb.columns}
            for c in df_wb.columns:
                if 'Country' in c and 'Name' in c:
                    col_map[c] = 'Country'
                elif 'Country' in c and 'Code' not in c: # Avoid Country Code
                    col_map[c] = 'Country'
            
            df_wb = df_wb.rename(columns=col_map)
            
            if 'Country' in df_wb.columns:
                # 找出年份欄位 (通常是數字 1960, 1961... 或字串 '1960')
                value_vars = [c for c in df_wb.columns if str(c).strip().isdigit()]
                
                # Melt 轉為長格式
                df_wb_melted = df_wb.melt(id_vars=['Country'], value_vars=value_vars, var_name='Year', value_name='GDP')
                df_wb_melted['Year'] = pd.to_numeric(df_wb_melted['Year'], errors='coerce')
                df_wb_melted['GDP'] = df_wb_melted['GDP'].apply(clean_currency)
                df_wb_melted = df_wb_melted.dropna(subset=['GDP', 'Year'])
            else:
                st.error("無法在資料中找到 'Country Name' 欄位，請檢查格式。")
                return pd.DataFrame()
        else:
            df_wb_melted = pd.DataFrame(columns=['Country', 'Year', 'GDP'])

    except Exception as e:
        st.error(f"資料讀取錯誤: {e}")
        return pd.DataFrame()

    # 3. 讀取內建台灣數據
    df_tw = pd.read_csv(io.StringIO(taiwan_data_csv))
    df_tw['Country'] = 'Taiwan'
    
    # 4. 合併
    if not df_wb_melted.empty:
        df_final = pd.concat([df_wb_melted, df_tw], ignore_index=True)
    else:
        df_final = df_tw # 如果沒上傳，至少顯示台灣
        
    df_final['Year'] = df_final['Year'].astype(int)
    df_final = df_final.sort_values(['Country', 'Year'])
    
    return df_final

# -----------------------------------------------------------------------------
# 3. 網頁介面 (Streamlit App)
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Economic Analysis (Full World)")

st.title("全球經濟增長與收斂分析 (World Data)")
st.markdown("""
本工具將**您提供的 World Bank 數據**與**內建台灣數據**合併分析。
請在下方上傳檔案或貼上文字資料以包含全世界國家。
""")

# --- Sidebar ---
st.sidebar.header("1. 資料來源 (Data Source)")
data_source_option = st.sidebar.radio(
    "選擇資料輸入方式:",
    ("貼上文字資料 (Paste Text)", "上傳檔案 (Upload File)", "僅使用內建範例 (Demo)")
)

df = pd.DataFrame()

if data_source_option == "貼上文字資料 (Paste Text)":
    raw_text = st.sidebar.text_area("請在此貼上 World Bank 數據 (包含 Country Name 及年份欄位):", height=200)
    if raw_text:
        df = process_data(pasted_text=raw_text)
    else:
        # 如果還沒貼，先用內建台灣+主要國家範例，以免報錯
        df = process_data() 
        st.sidebar.info("等待貼上資料... 目前顯示僅含台灣。")

elif data_source_option == "上傳檔案 (Upload File)":
    uploaded_file = st.sidebar.file_uploader("上傳 CSV 或 Excel 檔", type=['csv', 'xlsx'])
    if uploaded_file:
        df = process_data(uploaded_file=uploaded_file)
    else:
        df = process_data()
        st.sidebar.info("等待上傳檔案... 目前顯示僅含台灣。")

else: # Demo
    # 這裡為了方便，我把您上次提供的一小部分關鍵國家做成默認字串，讓Demo不至於只有台灣
    demo_csv = """Country	1995	2000	2005	2010	2015	2020	2023
Czechia	11453	12543	15147	17190	17931	19233	20251
Singapore	30379	34890	41798	48752	55645	59189	66167
Korea, Rep.	13411	16995	21197	25455	28737	31378	34121
United States	41710	48616	52649	52812	56849	59484	65505
China	1545	2237	3465	5764	8679	10573	12484
Japan	30171	31430	33098	32942	34960	34642	36952"""
    df = process_data(pasted_text=demo_csv)

# 確保數據不為空
if df.empty:
    st.error("無數據可顯示。")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.header("2. 篩選條件 (Filters)")

all_countries = sorted(df['Country'].unique())
# 預設選取
default_list = ['Taiwan', 'Korea, Rep.', 'Singapore', 'Czechia', 'United States', 'China']
valid_default = [c for c in default_list if c in all_countries]
if not valid_default and all_countries:
    valid_default = [all_countries[0]]

selected_countries = st.sidebar.multiselect(
    "選擇國家 (Select Countries)",
    all_countries,
    default=valid_default
)

# 時間滑桿 (動態範圍)
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
start_year, end_year = st.sidebar.slider(
    "年份範圍 (Year Range)",
    min_year, max_year, (1995, 2024)
)

if not selected_countries:
    st.warning("請至少選擇一個國家。")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["1. 經濟成長分析", "2. 收斂分析 (Convergence)", "3. 原始數據檢查"])

# --- TAB 1: Growth ---
with tab1:
    st.header("人均 GDP 趨勢 (Constant 2015 US$)")
    
    df_chart = df[
        (df['Country'].isin(selected_countries)) & 
        (df['Year'] >= start_year) & 
        (df['Year'] <= end_year)
    ]
    
    fig = px.line(df_chart, x='Year', y='GDP', color='Country', markers=True,
                  title="GDP per capita Trend")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("單一國家模型擬合 (Model Fitting)")
    
    c_model = st.selectbox("選擇要分析的國家:", selected_countries)
    df_s = df[(df['Country'] == c_model) & (df['Year'] >= start_year) & (df['Year'] <= end_year)].sort_values('Year')
    
    if len(df_s) > 3:
        X = df_s['Year'].values.reshape(-1, 1)
        y = df_s['GDP'].values
        t = X - X.min()
        
        # Models
        # 1. Linear
        model_lin = LinearRegression().fit(t, y)
        y_lin = model_lin.predict(t)
        r2_lin = r2_score(y, y_lin)
        
        # 2. Quadratic
        X_q = np.hstack([t, t**2])
        model_q = LinearRegression().fit(X_q, y)
        y_q = model_q.predict(X_q)
        r2_q = r2_score(y, y_q)
        
        # 3. Solow-Swan (Log-Linear)
        # Handle zeros/negative for log
        valid_idx = y > 0
        if np.any(valid_idx):
            y_log = np.log(y[valid_idx])
            t_valid = t[valid_idx]
            model_exp = LinearRegression().fit(t_valid, y_log)
            y_exp = np.exp(model_exp.predict(t)) # predict on full t might fail if shape differs, simplified here
            # Re-predict on full t for plotting
            y_exp_plot = np.exp(model_exp.predict(t))
            r2_exp = r2_score(y, y_exp_plot)
        else:
            y_exp_plot = y * 0
            r2_exp = 0

        # Plot
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df_s['Year'], y=y, mode='markers', name='實際數據', marker=dict(color='black')))
        fig_m.add_trace(go.Scatter(x=df_s['Year'], y=y_lin, mode='lines', name=f'線性 (R²={r2_lin:.3f})', line=dict(dash='dash', color='blue')))
        fig_m.add_trace(go.Scatter(x=df_s['Year'], y=y_q, mode='lines', name=f'二次方 (R²={r2_q:.3f})', line=dict(color='red')))
        fig_m.add_trace(go.Scatter(x=df_s['Year'], y=y_exp_plot, mode='lines', name=f'指數 (Solow) (R²={r2_exp:.3f})', line=dict(color='green')))
        
        st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.write("數據點不足以建立模型。")

# --- TAB 2: Convergence ---
with tab2:
    st.header("收斂分析 (Convergence)")
    
    # 1. Beta
    st.subheader("1. Beta Convergence (Catch-up Effect)")
    st.markdown("檢驗：初始 GDP 較低的國家，成長率是否較高？(斜率應為負)")
    
    beta_list = []
    for c in selected_countries:
        c_df = df[(df['Country'] == c) & (df['Year'] >= start_year) & (df['Year'] <= end_year)].sort_values('Year')
        if not c_df.empty:
            try:
                y_start = c_df.iloc[0]['GDP']
                y_end = c_df.iloc[-1]['GDP']
                duration = c_df.iloc[-1]['Year'] - c_df.iloc[0]['Year']
                
                if duration > 0 and y_start > 0 and y_end > 0:
                    g_rate = (np.log(y_end) - np.log(y_start)) / duration
                    beta_list.append({
                        'Country': c,
                        'ln_Initial_GDP': np.log(y_start),
                        'Avg_Growth_Rate': g_rate
                    })
            except:
                continue
                
    if len(beta_list) > 1:
        df_beta = pd.DataFrame(beta_list)
        
        # Reg
        X_b = df_beta['ln_Initial_GDP'].values.reshape(-1, 1)
        y_b = df_beta['Avg_Growth_Rate'].values
        mod_b = LinearRegression().fit(X_b, y_b)
        slope_b = mod_b.coef_[0]
        
        fig_b = px.scatter(df_beta, x='ln_Initial_GDP', y='Avg_Growth_Rate', text='Country',
                           title=f"Beta Convergence (Slope: {slope_b:.5f})",
                           labels={'ln_Initial_GDP': 'ln(Initial GDP)', 'Avg_Growth_Rate': 'Average Growth Rate'})
        
        # Trendline
        x_rng = np.linspace(df_beta['ln_Initial_GDP'].min(), df_beta['ln_Initial_GDP'].max(), 100).reshape(-1, 1)
        y_prd = mod_b.predict(x_rng)
        fig_b.add_trace(go.Scatter(x=x_rng.flatten(), y=y_prd, mode='lines', name='Reg Line', line=dict(color='red')))
        
        st.plotly_chart(fig_b, use_container_width=True)
        
        if slope_b < 0:
            st.success(f"✅ 斜率為負 ({slope_b:.5f})，存在收斂現象 (Convergence)。")
        else:
            st.error(f"❌ 斜率為正 ({slope_b:.5f})，存在發散現象 (Divergence)。")
            
    else:
        st.warning("選取的國家或數據不足以計算 Beta 收斂。")

    st.divider()
    
    # 2. Sigma
    st.subheader("2. Sigma Convergence (CV)")
    st.markdown("檢驗：各國人均 GDP 的變異係數 (CV) 是否隨時間下降？")
    
    cv_data = []
    years = range(start_year, end_year + 1)
    for y in years:
        sub = df[(df['Country'].isin(selected_countries)) & (df['Year'] == y)]
        if len(sub) > 1:
            mu = sub['GDP'].mean()
            sigma = sub['GDP'].std()
            if mu > 0:
                cv_data.append({'Year': y, 'CV': sigma / mu})
    
    if cv_data:
        df_cv = pd.DataFrame(cv_data)
        fig_c = px.line(df_cv, x='Year', y='CV', markers=True, title="Sigma Convergence (Coefficient of Variation)")
        st.plotly_chart(fig_c, use_container_width=True)
        st.caption("CV 下降代表貧富差距縮小 (收斂)。")

# --- TAB 3: Raw Data ---
with tab3:
    st.subheader("目前使用的合併數據")
    st.dataframe(df)