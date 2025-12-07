import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io

# ==========================================
# 1. 資料準備 (Data Preparation)
# ==========================================
# 使用您提供的 Excel 數據
csv_data = """Year,Korea,Singapore,Czech Republic,Taiwan
2005,"21,197.20","41,798.49","15,147.72","16,600.12"
2006,"22,196.21","44,159.41","16,107.30","17,485.93"
2007,"23,365.14","46,178.81","16,892.54","18,607.11"
2008,"23,887.06","44,601.58","17,190.65","18,690.28"
2009,"23,952.81","43,331.87","16,272.81","18,324.38"
2010,"25,455.62","48,752.05","16,665.93","20,147.16"
2011,"26,191.64","50,713.53","16,926.28","20,839.84"
2012,"26,680.28","51,679.35","16,772.35","21,232.32"
2013,"27,399.62","53,298.97","16,759.78","21,690.29"
2014,"28,100.01","54,681.93","17,118.04","22,656.57"
2015,"28,737.44","55,645.61","17,931.60","22,930.45"
2016,"29,467.12","56,986.79","18,359.10","23,374.06"
2017,"30,312.89","59,485.30","19,257.69","24,189.24"
2018,"31,059.27","61,250.35","19,736.63","24,866.13"
2019,"31,645.95","61,345.53","20,360.06","25,608.55"
2020,"31,378.16","59,189.70","19,233.15","26,499.99"
2021,"32,771.07","67,731.26","20,373.88","28,418.13"
2022,"33,690.38","68,218.81","20,627.35","29,366.31"
2023,"34,121.02","66,167.36","20,251.75","29,656.23
2024,"34,871.68","67,706.83","20,444.54","31,127.85"
"""

@st.cache_data
def load_data():
    df = pd.read_csv(io.StringIO(csv_data))
    # 清理數據：移除逗號並轉為浮點數
    for col in df.columns:
        if col != 'Year':
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

df = load_data()

# 轉換為長格式以便於處理
df_melted = df.melt(id_vars=['Year'], var_name='Country', value_name='GDP')

# ==========================================
# 2. 介面設定
# ==========================================
st.title("S-S 模型與經濟收斂分析 (2005-2024)")
st.markdown("""
本分析基於 Korea, Singapore, Czech Republic, Taiwan 四國的人均 GDP 數據。
根據講義定義：
- **S-S 模型係數**: 透過 $\ln(y_t) = \ln(y_0) + g \cdot t$ 計算，斜率 $g$ 即為成長係數。
""")

tab1, tab2 = st.tabs(["單一國家 S-S 模型分析", "收斂分析 (Convergence & Sigma)"])

# ==========================================
# Tab 1: 單一國家 S-S Model 分析
# ==========================================
with tab1:
    st.header("單一國家 S-S 模型參數估計")
    country = st.selectbox("選擇國家:", df.columns[1:])
    
    # 準備數據
    data_country = df[['Year', country]].copy()
    data_country.columns = ['Year', 'GDP']
    
    # 1. 數據轉換: 取對數 ln(GDP)
    data_country['ln_GDP'] = np.log(data_country['GDP'])
    
    # 2. 建立時間軸 t (從 0 開始)
    # 2005是第0年, 2006是第1年...
    data_country['t'] = data_country['Year'] - data_country['Year'].min()
    
    # 3. 線性回歸: ln(GDP) = intercept + slope * t
    X = data_country[['t']]
    y = data_country['ln_GDP']
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]      # 這就是 Koeficient (係數)
    intercept = model.intercept_ # 這就是 ln(Y0)
    r2 = r2_score(y, model.predict(X))
    
    # 4. 產生預測曲線 (還原為指數形式)
    # y = exp(intercept) * exp(slope * t)
    data_country['S-S Prediction'] = np.exp(intercept) * np.exp(slope * data_country['t'])
    
    # 顯示結果
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Koeficient (係數/成長率)", value=f"{slope:.4f} ({slope:.2%})")
        st.write(f"**回歸公式 (對數):** $\ln(y) = {intercept:.4f} + {slope:.4f} t$")
        st.write(f"**S-S 模型公式 (指數):** $y = {np.exp(intercept):.2f} \cdot e^{{{slope:.4f} t}}$")
        st.write(f"**R² (擬合度):** {r2:.4f}")
    
    with col2:
        st.dataframe(data_country[['Year', 'GDP', 'ln_GDP']].head())

    # 繪圖
    fig = go.Figure()
    # 實際數據點
    fig.add_trace(go.Scatter(x=data_country['Year'], y=data_country['GDP'], 
                             mode='markers',