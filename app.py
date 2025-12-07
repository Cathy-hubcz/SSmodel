Role: Expert Python Developer (Streamlit & Data Science)
Task: Create a single-file Streamlit app (`app.py`) for Economic Growth & Convergence Analysis.

# 1. Data Handling Logic (Most Important)
The user has a massive text dataset (Tab-separated values) with thousands separators (e.g., "11,900.58").
You must write a function `load_data()` that:
1. Defines a variable `raw_data_str = """..."""` (Leave a placeholder comment inside).
2. Uses `pd.read_csv(io.StringIO(raw_data_str), sep='\t', thousands=',')` to read it.
3. Cleans the data:
   - Drop columns that are not 'Country Name' or years (1970-2024).
   - Convert all Year columns to numeric (coerce errors).
4. **CRITICAL:** Manually append this Taiwan data row to the DataFrame (since it's missing in World Bank data):
   - Country Name: "Taiwan"
   - Data: {1970: 396, 1980: 2367, 1990: 8205, 2000: 14908, 2010: 19197, 2020: 28383, 2024: 34432} (Perform linear interpolation for missing years).
5. Reshapes the data (Melt) to columns: `['Country', 'Year', 'GDP']`.

# 2. App Interface (Streamlit)
**Sidebar:**
- Multi-select for Countries (Default: Taiwan, United States, China, Japan, Korea, Rep.).
- Slider for Year Range.

**Tab 1: Single Country (Solow-Swan)**
- Dropdown to select 1 country.
- Plot Scatter of actual GDP.
- Plot 3 Regression Lines:
  1. Linear ($y=ax+b$)
  2. Quadratic ($y=ax^2+bx+c$) - Label as "Slowing Growth?"
  3. Exponential/Log ($ln(y)=ax+b$) - Label as "S-S Model Path"
- Display R-squared for all three.

**Tab 2: Convergence (Beta & Sigma)**
- Use the countries selected in Sidebar.
- **Beta Convergence:** Plot [X=Initial ln(GDP), Y=Avg Growth Rate]. Add regression line. If slope is negative, display "Convergence Detected! (Catch-up effect)".
- **Sigma Convergence:** Plot Standard Deviation of ln(GDP) over time.
- **CV Plot:** Plot Coefficient of Variation (StdDev/Mean) over time.

# 3. Output
- Provide the **FULL Python code** in one block.
- **VERY IMPORTANT:** Inside the code, create the variable `raw_data_str` with triple quotes `"""` and write `# PASTE_YOUR_DATA_HERE` inside it, so the user can paste their massive dataset.