1242.46	1313.24	1379.79	1455.55	1566.20	1638.11	1692.83	1688.57	1689.54	1652.76	1587.20	1571.42	1638.62	1677.30	1767.07	1797.26
"""

# ==========================================
# 3. Data Processing Logic
# ==========================================

def clean_currency(x):
    """Cleans currency strings by removing commas and converting to float."""
    if isinstance(x, str):
        # Remove thousands separators and whitespace
        clean_str = x.replace(',', '').strip()
        return pd.to_numeric(clean_str, errors='coerce')
    return x

@st.cache_data
def load_data():
    """Loads and merges Taiwan data and World Bank data."""
    
    # 1. Process Taiwan Data
    # Use the placeholder CSV defined at the top of the file
    try:
        df_tw = pd.read_csv(io.StringIO(taiwan_data_csv))
        df_tw['Country'] = 'Taiwan'
        
        # Calculate ln(base index) for S-S model if needed: ln(GDP_t / GDP_1971)
        df_tw = df_tw.sort_values('Year')
        # Find 1971 data if available, otherwise use the first available year
        base_data = df_tw[df_tw['Year'] == 1971]
        if not base_data.empty:
            gdp_base = base_data['GDP'].iloc[0]
            if gdp_base > 0:
                df_tw['ln_Base_Index'] = np.log(df_tw['GDP'] / gdp_base)
            else:
                df_tw['ln_Base_Index'] = np.nan
        else:
            # Fallback if 1971 is missing
            df_tw['ln_Base_Index'] = np.nan
            
    except Exception as e:
        st.error(f"Error parsing Taiwan data: {e}")
        # Create an empty dataframe structure if parsing fails
        df_tw = pd.DataFrame(columns=['Year', 'GDP', 'Country', 'ln_Base_Index'])

    # 2. Process World Bank Data
    try:
        if len(raw_data_str.strip()) > 50:
            # Try reading as Tab-separated (standard World Bank format)
            df_wb = pd.read_csv(io.StringIO(raw_data_str), sep='\t', engine='python')
            
            # If that fails (only 1 column detected), try Comma-separated
            if df_wb.shape[1] < 2:
                df_wb = pd.read_csv(io.StringIO(raw_data_str), sep=',', engine='python')

            # Identify Country column
            col_map = {c: 'Country' for c in df_wb.columns if 'Country' in str(c)}
            df_wb = df_wb.rename(columns=col_map)
            
            # Identify Year columns (numeric headers)
            year_cols = [c for c in df_wb.columns if str(c).strip().isdigit()]
            
            # Melt: Convert Wide format (Years as columns) to Long format (Year rows)
            df_melted = df_wb.melt(id_vars=['Country'], value_vars=year_cols, var_name='Year', value_name='GDP')
            
            # Clean data types
            df_melted['Year'] = pd.to_numeric(df_melted['Year'], errors='coerce')
            df_melted['GDP'] = df_melted['GDP'].apply(clean_currency)
            
            # Drop invalid rows
            df_melted = df_melted.dropna(subset=['GDP', 'Year'])
            
            # 3. Combine Datasets
            df_final = pd.concat([df_melted, df_tw], ignore_index=True)
        else:
            df_final = df_tw
            
    except Exception as e:
        st.error(f"Error parsing World Bank data: {e}")
        return df_tw

    # Final cleanup
    df_final['Year'] = df_final['Year'].astype(int)
    df_final = df_final.sort_values(['Country', 'Year'])
    return df_final

# Load data once
df = load_data()

# ==========================================
# 4. Streamlit Interface
# ==========================================

# Sidebar Settings
st.sidebar.header("Nastavení (Settings)")

# Get unique countries
all_countries = sorted(df['Country'].unique())

# Set default selection
default_sel = ['Taiwan', 'Korea, Rep.', 'Czechia', 'Singapore', 'United States']
valid_sel = [c for c in default_sel if c in all_countries]
if not valid_sel and all_countries:
    valid_sel = [all_countries[0]]

selected_countries = st.sidebar.multiselect(
    "Vyberte země (Select Countries):",
    all_countries,
    default=valid_sel
)

# Year Slider
if not df.empty:
    min_y = int(df['Year'].min())
    max_y = int(df['Year'].max())
else:
    min_y, max_y = 1970, 2024

year_range = st.sidebar.slider("Rozsah let (Year Range):", min_y, max_y, (min_y, max_y))

# Filter Data
if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()

df_filtered = df[
    (df['Country'].isin(selected_countries)) & 
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1])
]

# --- Main Layout ---
st.title("Analýza hospodářského růstu a konvergence")
st.markdown("### Economic Growth and Convergence Analysis")

tab1, tab2 = st.tabs(["Jedna země (Single Country)", "Konvergence (Convergence)"])

# ==========================================
# TAB 1: Single Country Analysis
# ==========================================
with tab1:
    st.header("Analýza trendů HDP per capita")
    
    # Dropdown specifically for this tab
    c_single = st.selectbox("Vyberte zemi pro detailní analýzu:", selected_countries)
    
    # Get specific country data
    df_s = df_filtered[df_filtered['Country'] == c_single].sort_values('Year')
    
    if len(df_s) > 2:
        X_years = df_s['Year'].values
        y_gdp = df_s['GDP'].values
        
        # Normalize time t starting from 0 for regression stability
        # t = 0, 1, 2... corresponding to Year_min, Year_min+1...
        t = X_years - X_years.min()
        t_reshaped = t.reshape(-1, 1)

        # --- Model 1: Linear (y = ax + b) ---
        mod_lin = LinearRegression().fit(t_reshaped, y_gdp)
        y_lin = mod_lin.predict(t_reshaped)
        r2_lin = r2_score(y_gdp, y_lin)
        
        sign_lin = "+" if mod_lin.intercept_ >= 0 else "-"
        eq_lin = f"y = {mod_lin.coef_[0]:.2f}t {sign_lin} {abs(mod_lin.intercept_):.2f}"
        
        # --- Model 2: Quadratic (y = ax^2 + bx + c) ---
        poly = PolynomialFeatures(2)
        X_poly = poly.fit_transform(t_reshaped)
        mod_quad = LinearRegression().fit(X_poly, y_gdp)
        y_quad = mod_quad.predict(X_poly)
        r2_quad = r2_score(y_gdp, y_quad)
        
        # Coefficients: [intercept, linear, quadratic]
        c0 = mod_quad.intercept_
        c1 = mod_quad.coef_[1]
        c2 = mod_quad.coef_[2]
        
        sign_q1 = "+" if c1 >= 0 else "-"
        sign_q0 = "+" if c0 >= 0 else "-"
        eq_quad = f"y = {c2:.2f}t\u00b2 {sign_q1} {abs(c1):.2f}t {sign_q0} {abs(c0):.2f}"
        
        # --- Model 3: Exponential / S-S (y = A * e^(gt)) ---
        # Using Log-Linear transformation: ln(y) = ln(A) + gt
        
        # Safe log (handle zeros or negative GDP if any)
        valid_idx = y_gdp > 0
        if np.any(valid_idx):
            y_gdp_log = y_gdp[valid_idx]
            t_log = t_reshaped[valid_idx]
            X_years_log = X_years[valid_idx]
            
            y_log_val = np.log(y_gdp_log)
            mod_exp = LinearRegression().fit(t_log, y_log_val)
            
            # Predict
            y_log_pred = mod_exp.predict(t_log)
            y_exp = np.exp(y_log_pred) # Convert back to normal scale
            r2_exp = r2_score(y_gdp_log, y_exp)
            
            slope_g = mod_exp.coef_[0]      # Growth rate g
            intercept_lnA = mod_exp.intercept_ # ln(A)
            A_val = np.exp(intercept_lnA)
            
            sign_exp = "+" if intercept_lnA >= 0 else "-"
            eq_exp_log = f"ln(y) = {slope_g:.4f}t {sign_exp} {abs(intercept_lnA):.4f}"
            eq_exp_real = f"y = {A_val:.2f}e^({slope_g:.4f}t)"
        else:
            y_exp = np.zeros_like(y_gdp)
            r2_exp = 0
            eq_exp_log = "N/A"
            eq_exp_real = "N/A"
            X_years_log = X_years # fallback for plotting

        # --- Plotting ---
        fig = go.Figure()
        
        # Actual Data
        fig.add_trace(go.Scatter(
            x=X_years, y=y_gdp, mode='markers', name='Skutečná data (Actual)',
            marker=dict(size=8, color='blue')
        ))
        
        # Linear
        fig.add_trace(go.Scatter(
            x=X_years, y=y_lin, mode='lines', name=f'Lineární (R\u00b2={r2_lin:.3f})',
            line=dict(dash='dash', color='green')
        ))
        
        # Quadratic
        fig.add_trace(go.Scatter(
            x=X_years, y=y_quad, mode='lines', name=f'Kvadratický (R\u00b2={r2_quad:.3f})',
            line=dict(color='orange')
        ))
        
        # Exponential
        if np.any(valid_idx):
            fig.add_trace(go.Scatter(
                x=X_years_log, y=y_exp, mode='lines', name=f'Exponenciální (R\u00b2={r2_exp:.3f})',
                line=dict(dash='dot', color='red')
            ))

        fig.update_layout(title=f"HDP per capita: {c_single}", xaxis_title="Rok", yaxis_title="GDP")
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Statistics ---
        st.markdown("#### Rovnice modelů (Equations)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Lineární**\n\n${eq_lin}$")
        with col2:
            st.warning(f"**Kvadratický**\n\n${eq_quad}$")
        with col3:
            st.error(f"**Exponenciální (S-S)**\n\n${eq_exp_real}$")
            
    else:
        st.error("Nedostatek dat pro zvolenou zemi a období.")

# ==========================================
# TAB 2: Convergence Analysis
# ==========================================
def plot_trendline(df_in, x_col, y_col, title, label_y):
    """Helper to calculate trendline and return figure and slope."""
    if len(df_in) < 2:
        return None, 0
    
    X = df_in[x_col].values.reshape(-1, 1)
    y = df_in[y_col].values
    
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    eq = f"y = {slope:.5f}x + {intercept:.2f}"
    
    fig = px.scatter(df_in, x=x_col, y=y_col, title=f"{title} (Slope: {slope:.5f})", 
                     text=df_in.get('Country', None)) # Use country labels if available
    
    # Add trendline trace
    fig.add_trace(go.Scatter(
        x=df_in[x_col], y=y_pred, mode='lines', name='Trend', line=dict(color='red')
    ))
    
    # Annotation
    fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=f"{eq}<br>R\u00b2={r2:.3f}", showarrow=False,
        xanchor='left', bgcolor="white", opacity=0.8
    )
    
    return fig, slope

with tab2:
    st.header("Analýza konvergence")
    
    # ---------------------------
    # 1. Beta Convergence
    # ---------------------------
    st.subheader("1. Beta Konvergence")
    st.markdown("Vztah mezi počátečním HDP (log) a průměrným růstem. **Záporný sklon = Konvergence**.")
    
    beta_data = []
    y_start = year_range[0]
    y_end = year_range[1]
    
    for c in selected_countries:
        c_df = df[df['Country'] == c]
        
        # Get start and end values safely
        row_start = c_df[c_df['Year'] == y_start]
        row_end = c_df[c_df['Year'] == y_end]
        
        if not row_start.empty and not row_end.empty:
            gdp_0 = row_start.iloc[0]['GDP']
            gdp_T = row_end.iloc[0]['GDP']
            
            if gdp_0 > 0 and gdp_T > 0:
                # Calculate Average Annual Growth Rate approx: (ln(yT) - ln(y0)) / T
                T = y_end - y_start
                if T > 0:
                    growth = (np.log(gdp_T) - np.log(gdp_0)) / T
                    beta_data.append({
                        'Country': c,
                        'ln_GDP_initial': np.log(gdp_0),
                        'Avg_Growth': growth
                    })
    
    if len(beta_data) > 1:
        df_beta = pd.DataFrame(beta_data)
        fig_beta, slope_beta = plot_trendline(df_beta, 'ln_GDP_initial', 'Avg_Growth', 'Beta Konvergence', 'Growth')
        
        if fig_beta:
            st.plotly_chart(fig_beta, use_container_width=True)
            if slope_beta < 0:
                st.success(f"Sklon je {slope_beta:.5f} (Záporný) -> Konvergence potvrzena.")
            else:
                st.error(f"Sklon je {slope_beta:.5f} (Kladný) -> Divergence.")
    else:
        st.warning("Nedostatek dat pro výpočet Beta konvergence (potřeba min. 2 země s daty v počátečním i koncovém roce).")

    st.markdown("---")

    # ---------------------------
    # 2. Sigma Convergence
    # ---------------------------
    st.subheader("2. Sigma Konvergence (Std Dev logaritmů)")
    st.markdown("Vývoj směrodatné odchylky logaritmů HDP v čase. **Klesající trend = Konvergence**.")
    
    sigma_data = []
    # Loop through every year in the selected range
    for yr in range(year_range[0], year_range[1] + 1):
        # Slice data for that year across selected countries
        sub = df_filtered[df_filtered['Year'] == yr]
        
        # We need at least 2 countries to calculate std dev
        if len(sub) > 1:
            # Calculate std dev of natural log of GDP
            valid_gdp = sub[sub['GDP'] > 0]['GDP']
            if len(valid_gdp) > 1:
                std_log = np.std(np.log(valid_gdp))
                sigma_data.append({'Year': yr, 'Sigma': std_log})
                
    if len(sigma_data) > 1:
        df_sigma = pd.DataFrame(sigma_data)
        # We can treat this as a line chart
        fig_sigma = px.line(df_sigma, x='Year', y='Sigma', markers=True, title="Sigma Konvergence")
        
        # Add a simple trendline for visual aid
        X_sig = df_sigma['Year'].values.reshape(-1, 1)
        y_sig = df_sigma['Sigma'].values
        mod_sig = LinearRegression().fit(X_sig, y_sig)
        y_sig_pred = mod_sig.predict(X_sig)
        
        fig_sigma.add_trace(go.Scatter(x=df_sigma['Year'], y=y_sig_pred, mode='lines', name='Trend', line=dict(dash='dash', color='red')))
        
        st.plotly_chart(fig_sigma, use_container_width=True)
        
        slope_sigma = mod_sig.coef_[0]
        if slope_sigma < 0:
            st.success(f"Trend Sigmy je klesající ({slope_sigma:.5f}) -> Konvergence.")
        else:
            st.error(f"Trend Sigmy je rostoucí ({slope_sigma:.5f}) -> Divergence.")
            
    else:
        st.warning("Nedostatek dat pro Sigma konvergenci.")

    st.markdown("---")

    # ---------------------------
    # 3. CV Convergence
    # ---------------------------
    st.subheader("3. Variační Koeficient (CV)")
    st.markdown("CV = Směrodatná odchylka / Průměr. **Klesající trend = Konvergence**.")
    
    cv_data = []
    for yr in range(year_range[0], year_range[1] + 1):
        sub = df_filtered[df_filtered['Year'] == yr]
        if len(sub) > 1:
            mean_gdp = sub['GDP'].mean()
            std_gdp = sub['GDP'].std()
            
            if mean_gdp > 0:
                cv = std_gdp / mean_gdp
                cv_data.append({'Year': yr, 'CV': cv})
                
    if len(cv_data) > 1:
        df_cv = pd.DataFrame(cv_data)
        fig_cv = px.line(df_cv, x='Year', y='CV', markers=True, title="Vývoj Variačního Koeficientu")
        
        # Trendline
        X_cv = df_cv['Year'].values.reshape(-1, 1)
        y_cv = df_cv['CV'].values
        mod_cv = LinearRegression().fit(X_cv, y_cv)
        y_cv_pred = mod_cv.predict(X_cv)
        
        fig_cv.add_trace(go.Scatter(x=df_cv['Year'], y=y_cv_pred, mode='lines', name='Trend', line=dict(dash='dash', color='red')))
        
        st.plotly_chart(fig_cv, use_container_width=True)
        
        slope_cv = mod_cv.coef_[0]
        if slope_cv < 0:
            st.success(f"Trend CV je klesající ({slope_cv:.5f}) -> Konvergence.")
        else:
            st.error(f"Trend CV je rostoucí ({slope_cv:.5f}) -> Divergence.")
    else:
        st.warning("Nedostatek dat pro CV analýzu.")