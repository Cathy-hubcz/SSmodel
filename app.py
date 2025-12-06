import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import io

# ==========================================
# 1. 台灣數據 (Taiwan Data - Hardcoded)
# ==========================================
# 這是您提供的精確數據 (1971-2024)
taiwan_data_str = """Year,GDP
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

# ==========================================
# 2. 資料處理函數
# ==========================================
def process_data(user_text_data):
    """
    處理使用者貼上的 World Bank 資料與內建台灣資料。
    """
    # 1. 處理台灣資料
    df_tw = pd.read_csv(io.StringIO(taiwan_data_str))
    df_tw['Country'] = 'Taiwan'
    
    # 2. 處理使用者貼上的 World Bank 資料 (如果是空的，只回傳台灣)
    if not user_text_data or len(user_text_data.strip()) < 10:
        return df_tw

    try:
        # 嘗試讀取 Tab 分隔資料 (World Bank 預設格式)
        # 關鍵：thousands=',' 處理 "11,900.58" 這種格式
        df_wb = pd.read_csv(io.StringIO(user_text_data), sep='\t', thousands=',')
        
        # 如果欄位太少，可能是用逗號分隔的 CSV
        if df_wb.shape[1] < 2:
            df_wb = pd.read_csv(io.StringIO(user_text_data), sep=',', thousands=',')

        # 清洗：找出包含 'Country' 的欄位並重新命名
        col_map = {c: 'Country' for c in df_wb.columns if 'Country' in str(c)}
        df_wb = df_wb.rename(columns=col_map)
        
        if 'Country' not in df_wb.columns:
            st.error("Chyba formátu dat: Sloupec 'Country' nenalezen. (資料格式錯誤：找不到 Country 欄位)")
            return df_tw

        # 找出年份欄位 (數字組成)
        year_cols = [c for c in df_wb.columns if str(c).strip().isdigit()]
        
        # Melt (轉置: 寬表變長表)
        df_melted = df_wb.melt(id_vars=['Country'], value_vars=year_cols, var_name='Year', value_name='GDP')
        
        # 轉換數值
        df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
        # 移除貨幣符號、逗號等非數字字元
        if df_melted['GDP'].dtype == object:
             df_melted['GDP'] = df_melted['GDP'].astype(str).str.replace(',', '').str.replace('"', '').str.replace('$', '')
        df_melted['GDP'] = pd.to_numeric(df_melted['GDP'], errors='coerce')
        
        df_melted = df_melted.dropna(subset=['GDP', 'Year'])
        
        # 合併台灣資料
        df_final = pd.concat([df_melted, df_tw], ignore_index=True)
        
        # 確保年份排序
        df_final = df_final.sort_values(['Country', 'Year'])
        return df_final

    except Exception as e:
        st.error(f"Error processing data: {e}")
        return df_tw

# ==========================================
# 3. Streamlit 介面設定
# ==========================================
st.set_page_config(layout="wide", page_title="Analýza růstu a konvergence")

st.title("Analýza hospodářského růstu a konvergence (Economic Growth & Convergence)")
st.markdown("---")

# --- SIDEBAR (Boční panel / 側邊欄) ---
st.sidebar.header("1. Vstup dat (Data Input / 資料輸入)")

# 定義 Raw Data String 變數，讓使用者貼上
# PASTE_YOUR_DATA_HERE
raw_data_placeholder = """Paste your World Bank Data (Tab-separated) here..."""
user_input = st.sidebar.text_area("Vložte data zde (Paste Excel/Text data):", height=150, help="貼上包含 Country 和年份的數據")

# 載入並處理資料
df_all = process_data(user_input)

# 取得所有國家列表
all_countries = sorted(df_all['Country'].unique())

# 預設選取國家
default_selection = ['Taiwan', 'Korea, Rep.', 'Czechia', 'Singapore']
# 過濾掉資料中不存在的預設國家
valid_defaults = [c for c in default_selection if c in all_countries]
# 如果預設的都沒找到，就選前三個
if not valid_defaults and len(all_countries) > 0:
    valid_defaults = all_countries[:3]

st.sidebar.header("2. Nastavení (Settings / 設定)")
selected_countries = st.sidebar.multiselect(
    "Vyberte země (Select Countries / 選擇國家):",
    options=all_countries,
    default=valid_defaults
)

# 年份滑桿
min_year = int(df_all['Year'].min())
max_year = int(df_all['Year'].max())
year_range = st.sidebar.slider(
    "Rozsah let (Year Range / 年份範圍):",
    min_value=min_year,
    max_value=max_year,
    value=(2005, 2024) # 預設改為您提到的 2005-2024 以符合 PPT
)

# 過濾資料
df_filtered = df_all[
    (df_all['Country'].isin(selected_countries)) & 
    (df_all['Year'] >= year_range[0]) & 
    (df_all['Year'] <= year_range[1])
]

if df_filtered.empty:
    st.warning("Žádná data k zobrazení. Zkontrolujte výběr. (無數據顯示，請檢查選擇)")
    st.stop()

# ==========================================
# 4. 主內容 (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["Jedna země (Single Country / 單一國家模型)", "Konvergence (Convergence / 收斂分析)"])

# --- TAB 1: Solow-Swan & Trends ---
with tab1:
    st.subheader("Analýza trendů HDP (GDP Trends / GDP 趨勢分析)")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        target_country = st.selectbox("Vyberte zemi pro detail (Select Country / 選擇單一國家):", selected_countries)
    
    # 準備單一國家資料
    df_one = df_filtered[df_filtered['Country'] == target_country].sort_values('Year')
    
    if len(df_one) > 2:
        X = df_one['Year'].values.reshape(-1, 1)
        y = df_one['GDP'].values
        
        # 1. 線性回歸 (Linear) y = ax + b
        model_lin = LinearRegression().fit(X, y)
        y_pred_lin = model_lin.predict(X)
        r2_lin = r2_score(y, y_pred_lin)
        eq_lin = f"y = {model_lin.coef_[0]:.2f}x + ({model_lin.intercept_:.2f})"

        # 2. 二次式回歸 (Quadratic) y = ax^2 + bx + c
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model_quad = LinearRegression().fit(X_poly, y)
        y_pred_quad = model_quad.predict(X_poly)
        r2_quad = r2_score(y, y_pred_quad)
        # coef_[2] is a (x^2), coef_[1] is b (x)
        eq_quad = f"y = {model_quad.coef_[2]:.4f}x^2 + {model_quad.coef_[1]:.2f}x + {model_quad.intercept_:.2f}"

        # 3. 指數回歸 (Exponential / Solow-Swan proxy) ln(y) = ax + b -> y = e^(ax+b)
        y_log = np.log(y)
        model_exp = LinearRegression().fit(X, y_log)
        y_pred_log = model_exp.predict(X)
        y_pred_exp = np.exp(y_pred_log)
        r2_exp = r2_score(y, y_pred_exp) # Calculate R2 on original scale
        eq_exp = f"ln(y) = {model_exp.coef_[0]:.4f}x + ({model_exp.intercept_:.4f})"

        # --- 繪圖 (Plotly) ---
        fig = go.Figure()
        # 實際數據
        fig.add_trace(go.Scatter(x=df_one['Year'], y=df_one['GDP'], mode='markers', name='Data (Skutečnost)', marker=dict(size=8, color='black')))
        # 線性
        fig.add_trace(go.Scatter(x=df_one['Year'], y=y_pred_lin, mode='lines', name=f'Linear (Lineární): R²={r2_lin:.4f}', line=dict(dash='dash', color='blue')))
        # 二次
        fig.add_trace(go.Scatter(x=df_one['Year'], y=y_pred_quad, mode='lines', name=f'Quadratic (Kvadratický): R²={r2_quad:.4f}', line=dict(color='red')))
        # 指數
        fig.add_trace(go.Scatter(x=df_one['Year'], y=y_pred_exp, mode='lines', name=f'Exponential (Exponenciální): R²={r2_exp:.4f}', line=dict(dash='dot', color='green')))

        fig.update_layout(
            title=f"Vývoj HDP: {target_country} (GDP Evolution / GDP 演變)",
            xaxis_title="Rok (Year)",
            yaxis_title="HDP na obyvatele (GDP per capita)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 顯示方程式與分析 ---
        st.markdown("#### Statistická analýza (Statistical Analysis / 統計分析)")
        st.info(
            f"""
            **Rovnice regresních modelů (Regression Equations / 回歸方程式):**
            
            1. **Lineární (Linear / 線性):** ${eq_lin}$ ($R^2 = {r2_lin:.4f}$)
            2. **Kvadratický (Quadratic / 二次式):** ${eq_quad}$ ($R^2 = {r2_quad:.4f}$)
               * (Poznámka: Kvadratický model lépe zachycuje zpomalení nebo zrychlení růstu. / Chinese: 二次式更能捕捉成長減緩或加速的趨勢)
            3. **Exponenciální (S-S Model Path):** ${eq_exp}$ ($R^2 = {r2_exp:.4f}$)
            """
        )

    else:
        st.write("Nedostatek dat pro regresi (Not enough data / 數據不足).")

# --- TAB 2: Convergence ---
with tab2:
    st.header("Analýza konvergence (Convergence Analysis / 收斂分析)")
    
    # 資料準備 for Beta Convergence
    # 邏輯：取每個國家在 "選定時間範圍" 的 "第一年" 與 "最後一年"
    convergence_data = []
    
    # 確保我們只使用選擇範圍內的數據
    start_year = year_range[0]
    end_year = year_range[1]
    
    st.caption(f"**Analyzované období (Period / 分析期間):** {start_year} - {end_year}")
    st.caption(f"**Vybrané země (Countries / 選擇國家):** {', '.join(selected_countries)}")

    for country in selected_countries:
        # 取出該國在範圍內的數據
        c_data = df_filtered[df_filtered['Country'] == country].sort_values('Year')
        
        # 必須要有頭尾年份的資料
        row_start = c_data[c_data['Year'] == start_year]
        row_end = c_data[c_data['Year'] == end_year]
        
        if not row_start.empty and not row_end.empty:
            y0 = row_start.iloc[0]['GDP']
            yT = row_end.iloc[0]['GDP']
            T = end_year - start_year
            
            if T > 0 and y0 > 0 and yT > 0:
                # 成長率計算公式: (ln(yT) - ln(y0)) / T
                growth_rate = (np.log(yT) - np.log(y0)) / T
                ln_initial = np.log(y0)
                convergence_data.append({
                    'Country': country,
                    'ln_Init': ln_initial,
                    'Growth': growth_rate,
                    'GDP_Start': y0,
                    'GDP_End': yT
                })
    
    df_conv = pd.DataFrame(convergence_data)

    # === Beta Convergence Plot ===
    col_beta, col_sigma = st.columns(2)
    
    with col_beta:
        st.subheader("β-Konvergence (Beta Convergence)")
        if len(df_conv) > 1:
            # 回歸
            X_beta = df_conv['ln_Init'].values.reshape(-1, 1)
            y_beta = df_conv['Growth'].values
            
            model_beta = LinearRegression().fit(X_beta, y_beta)
            slope = model_beta.coef_[0]
            intercept = model_beta.intercept_
            r2_beta = r2_score(y_beta, model_beta.predict(X_beta))
            
            eq_beta_str = f"y = {slope:.5f}x + {intercept:.5f}"
            
            # 判斷收斂
            convergence_text = "Konvergence (Convergence / 收斂) ✅" if slope < 0 else "Divergence (Rozbíhavost / 發散) ❌"
            color_res = "green" if slope < 0 else "red"

            # 繪圖
            fig_beta = px.scatter(df_conv, x='ln_Init', y='Growth', text='Country',
                                  labels={'ln_Init': 'ln(Počáteční HDP) / ln(Initial GDP)', 'Growth': 'Průměrný růst / Avg Growth'})
            
            # 畫回歸線
            x_range = np.linspace(df_conv['ln_Init'].min(), df_conv['ln_Init'].max(), 100).reshape(-1, 1)
            y_range = model_beta.predict(x_range)
            fig_beta.add_trace(go.Scatter(x=x_range.flatten(), y=y_range, mode='lines', name='Regression Line'))
            
            fig_beta.update_traces(textposition='top center')
            fig_beta.update_layout(showlegend=False)
            st.plotly_chart(fig_beta, use_container_width=True)
            
            st.markdown(f"**Výsledek (Result / 結果):** :{color_res}[{convergence_text}]")
            st.markdown(f"**Rovnice (Equation):** ${eq_beta_str}$")
            st.markdown(f"**R²:** {r2_beta:.4f}")
            st.caption("Poznámka: Záporný sklon (slope) znamená, že chudší země rostou rychleji. (中文註記: 斜率為負代表貧窮國家成長較快，存在追趕效應)")
            
        else:
            st.warning("Nedostatek zemí pro výpočet konvergence (alespoň 2). (需至少兩個國家)")

    # === Sigma Convergence Plot ===
    with col_sigma:
        st.subheader("σ-Konvergence (Sigma Convergence)")
        # 計算每年的 std(ln(GDP))
        sigma_data = []
        years_list = sorted(df_filtered['Year'].unique())
        
        for yr in years_list:
            sub = df_filtered[df_filtered['Year'] == yr]
            if len(sub) > 1:
                std_log = np.std(np.log(sub['GDP']))
                cv = np.std(sub['GDP']) / np.mean(sub['GDP']) # 變異係數
                sigma_data.append({'Year': yr, 'Sigma': std_log, 'CV': cv})
        
        df_sigma = pd.DataFrame(sigma_data)
        
        if not df_sigma.empty:
            fig_sigma = px.line(df_sigma, x='Year', y='Sigma', markers=True,
                                labels={'Sigma': 'Směrodatná odchylka ln(HDP) / StdDev ln(GDP)'})
            fig_sigma.update_layout(title="Nerovnost v čase (Inequality over Time)")
            st.plotly_chart(fig_sigma, use_container_width=True)
            st.caption("Klesající křivka znamená snižování nerovnosti mezi zeměmi. (中文註記: 曲線下降代表國家間貧富差距縮小)")
        else:
            st.write("Nedostatek dat.")
            
    # === Coefficient of Variation (CV) ===
    st.markdown("---")
    st.subheader("Variační koeficient (Coefficient of Variation /變異係數)")
    if not df_sigma.empty:
        fig_cv = px.line(df_sigma, x='Year', y='CV', markers=True, color_discrete_sequence=['orange'])
        fig_cv.update_layout(yaxis_title="CV (StdDev / Mean)")
        st.plotly_chart(fig_cv, use_container_width=True)