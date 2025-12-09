import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Havlena-Odeh Analysis", layout="wide")

st.title("📊 Havlena-Odeh Material Balance Analysis")

# ============================================================
# CONSTANTS
# ============================================================
CUFT_PER_BBL = 5.615

# ============================================================
# CHECK FOR DATA
# ============================================================

# Check if data is loaded
if "data_loaded" not in st.session_state or not st.session_state["data_loaded"]:
    st.warning("⚠️ No data loaded yet!")
    st.info("Please go to the **Upload Data** page first to load your production and PVT data.")
    st.stop()

# Get data from session state
production_data = st.session_state.get("production_data_std")
initial_params = st.session_state.get("initial_params", {})

if production_data is None:
    st.error("❌ Data not properly loaded. Please reload your data.")
    st.stop()

# Show what data we have
st.info(f"📁 Loaded {len(production_data)} data points")

# ============================================================
# DATA PREPARATION
# ============================================================

# Clean column names
production_data.columns = production_data.columns.str.strip()

# Display raw data for debugging
with st.expander("🔍 View Raw Uploaded Data"):
    st.write(production_data)

# Standardize column names
col_rename = {}
for col in production_data.columns:
    col_lower = col.lower()
    if 'pressure' in col_lower or col_lower == 'p' or col_lower == 'pressure':
        col_rename[col] = 'P'
    elif 'np' in col_lower or 'n_p' in col_lower:
        col_rename[col] = 'Np'
    elif 'gp' in col_lower or 'g_p' in col_lower:
        col_rename[col] = 'Gp'
    elif 'bo' in col_lower and 'bg' not in col_lower:
        col_rename[col] = 'Bo'
    elif 'rs' in col_lower or 'r_s' in col_lower:
        col_rename[col] = 'Rs'
    elif 'bg' in col_lower or 'b_g' in col_lower:
        col_rename[col] = 'Bg_cuft_scf'  # Store as cu ft/SCF initially

# Apply renaming
calc_df = production_data.rename(columns=col_rename).copy()

# Ensure we have all required columns
required_cols = ['P', 'Np', 'Gp', 'Bo', 'Rs', 'Bg_cuft_scf']
missing_cols = [col for col in required_cols if col not in calc_df.columns]

if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.info(f"Found columns: {list(calc_df.columns)}")
    st.stop()

# Convert to numeric
for col in required_cols:
    calc_df[col] = pd.to_numeric(calc_df[col], errors='coerce')

# Remove rows with missing values
calc_df = calc_df.dropna(subset=required_cols)

if len(calc_df) < 2:
    st.error("❌ Not enough valid data points. Need at least 2 rows.")
    st.stop()

# ============================================================
# UNIT CONVERSION - ALWAYS CONVERT Bg FROM cu ft/SCF TO bbl/SCF
# ============================================================

# Store original Bg values
calc_df['Bg_cuft_scf_original'] = calc_df['Bg_cuft_scf'].copy()

# Convert Bg from cu ft/SCF to bbl/SCF
calc_df['Bg_bbl_scf'] = calc_df['Bg_cuft_scf'] / CUFT_PER_BBL

# ============================================================
# GET OPTIONAL PARAMETERS (DEFAULT TO 0 IF NOT PROVIDED)
# ============================================================

# Water production (if column exists)
if 'Wp' not in calc_df.columns:
    calc_df['Wp'] = 0

# Water formation volume factor
Bw = 1.0

# Connate water saturation
Swc = initial_params.get('Swc', 0)

# Water compressibility
Cw = initial_params.get('Cw', 0)

# Formation compressibility
Cf = initial_params.get('Cf', 0)

# Water influx
We = 0

# ============================================================
# EXTRACT INITIAL CONDITIONS
# ============================================================

# Find initial conditions (minimum Np, usually 0)
initial_idx = calc_df['Np'].idxmin()
initial_row = calc_df.loc[initial_idx]

# Get initial parameters
pi = float(initial_row['P'])
Boi = float(initial_row['Bo'])
Rsi = float(initial_row['Rs'])
Bgi_cuft = float(initial_row['Bg_cuft_scf'])  # Initial Bg in cu ft/SCF
Bgi_bbl = Bgi_cuft / CUFT_PER_BBL  # Initial Bg in bbl/SCF



# ============================================================
# HAVLENA-ODEH CALCULATIONS (CORRECTED)
# ============================================================

st.header("🔧 Havlena-Odeh Calculations")

# Calculate Rp (producing GOR)
# IMPORTANT: Gp is in MMMSCF (billion), Np is in MMSTB (million)
# Rp = (Gp * 1000) / Np [SCF/STB]
calc_df['Rp'] = (calc_df['Gp']) / calc_df['Np']
calc_df['Rp'] = calc_df['Rp'].replace([np.inf, -np.inf], 0)

# Calculate ΔP
calc_df['ΔP'] = pi - calc_df['P']

# **F (Underground Withdrawal) in million bbl**
# F = Np[Bo + (Rp - Rs)Bg] + WpBw
# Where Np is in MMSTB, so F will be in MM bbl
# IMPORTANT: Use Bg in bbl/SCF
calc_df['F'] = calc_df['Np'] * (calc_df['Bo'] + (calc_df['Rp'] - calc_df['Rs']) * calc_df['Bg_bbl_scf']) + calc_df['Wp'] * Bw

# **Eo (Oil and Solution Gas Expansion) in bbl/STB**
# Eo = (Bo - Boi) + (Rsi - Rs)Bg
# Important: Bg MUST be in bbl/SCF for correct units
calc_df['Eo'] = (calc_df['Bo'] - Boi) + (Rsi - calc_df['Rs']) * calc_df['Bg_bbl_scf']

# **Eg (Gas-Cap Expansion) in bbl/STB**
# Eg = Boi * (Bg/Bgi - 1)
calc_df['Eg'] = Boi * (calc_df['Bg_bbl_scf'] / Bgi_bbl - 1)

# Calculate Efw (Formation and Water Expansion) in bbl/STB
# Efw = Boi * (Cw*Swc + Cf) * ΔP / (1 - Swc)
if Swc < 1:
    calc_df['Efw'] = Boi * (Cw * Swc + Cf) * calc_df['ΔP'] / (1 - Swc)
else:
    calc_df['Efw'] = 0

# Calculate ratios for straight-line analysis
# Handle division by zero or near-zero values
calc_df['F/Eo'] = np.where(
    np.abs(calc_df['Eo']) > 1e-10, 
    calc_df['F'] / calc_df['Eo'], 
    np.nan
)

calc_df['Eg/Eo'] = np.where(
    np.abs(calc_df['Eo']) > 1e-10, 
    calc_df['Eg'] / calc_df['Eo'], 
    np.nan
)

# Replace infinities with NaN
calc_df.replace([np.inf, -np.inf], np.nan, inplace=True)


# ============================================================
# DISPLAY CALCULATION TABLE
# ============================================================

st.subheader("📋 Calculated Values")

# Create display dataframe with units
display_df = pd.DataFrame()

# Add original data
display_df['P (psi)'] = calc_df['P'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else '---')
display_df['Np (MMSTB)'] = calc_df['Np'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else '---')
display_df['Gp (MMMSCF)'] = calc_df['Gp'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else '---')
display_df['Rp (SCF/STB)'] = calc_df['Rp'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '---')
display_df['Bo (bbl/STB)'] = calc_df['Bo'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '---')
display_df['Rs (SCF/STB)'] = calc_df['Rs'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else '---')
display_df['Bg (cu ft/SCF)'] = calc_df['Bg_cuft_scf_original'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else '---')
display_df['Bg (bbl/SCF)'] = calc_df['Bg_bbl_scf'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else '---')

# Add calculated values
display_df['F (MMbbl)'] = calc_df['F'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else '---')
display_df['Eo (bbl/STB)'] = calc_df['Eo'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else '---')
display_df['Eg (bbl/STB)'] = calc_df['Eg'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else '---')

if Cw != 0 or Cf != 0 or Swc != 0:
    display_df['Efw (bbl/STB)'] = calc_df['Efw'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else '---')

# Add ratios with careful formatting
def format_ratio(x):
    if pd.isna(x) or not np.isfinite(x):
        return '---'
    return f"{x:.2f}"

display_df['F/Eo'] = calc_df['F/Eo'].apply(format_ratio)
display_df['Eg/Eo'] = calc_df['Eg/Eo'].apply(format_ratio)

st.dataframe(display_df, use_container_width=True, height=400)

# ============================================================
# GAS CAP DRIVE ANALYSIS (Main Analysis)
# ============================================================

st.header("📈 Gas Cap Drive Analysis (F/Eo vs Eg/Eo)")

# Remove initial point (where Eo = 0) and invalid points
analysis_df = calc_df[calc_df['Eo'] > 1e-10].copy()  # Changed from >0 to >1e-10 for numerical stability
analysis_df = analysis_df.dropna(subset=['F/Eo', 'Eg/Eo'])
analysis_df = analysis_df[np.isfinite(analysis_df['F/Eo']) & np.isfinite(analysis_df['Eg/Eo'])]

if len(analysis_df) >= 2:
    x = analysis_df['Eg/Eo'].values
    y = analysis_df['F/Eo'].values
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r_value ** 2
    
    N_est = intercept
    m_est = slope / intercept if intercept != 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("N (OOIP)", f"{N_est:.2f} MMSTB")
    with col2:
        st.metric("m (gas cap ratio)", f"{m_est:.4f}")
    with col3:
        st.metric("R²", f"{r2:.4f}")
    
    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points',
        marker=dict(size=10, color='blue'),
        text=[f"P={p} psi" for p in analysis_df['P'].values]
    ))
    
    x_line = np.linspace(min(x)*0.9, max(x)*1.1, 100)
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=f'Regression: y = {slope:.3f}x + {intercept:.2f}',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Havlena-Odeh Straight Line: F/Eo vs Eg/Eo',
        xaxis_title='Eg/Eo',
        yaxis_title='F/Eo',
        height=500,
        showlegend=True
    )
    
    # Add equation annotation
    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"y = {slope:.2f}x + {intercept:.2f}<br>N = {intercept:.2f} MMSTB<br>m = {m_est:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================
    # DRIVE MECHANISM ANALYSIS
    # ============================================================
    
    st.subheader("🔍 Drive Mechanism Analysis")
    
    # Create analysis cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Quality", 
                  f"{'Excellent' if r2 > 0.95 else 'Good' if r2 > 0.85 else 'Poor'}",
                  f"{r2:.4f}")
    
    with col2:
        m_status = "✅ Positive" if m_est > 0.01 else "⚠️ Near Zero" if m_est >= 0 else "❌ Negative"
        st.metric("Gas Cap (m)", m_status, f"{m_est:.4f}")
    
    with col3:
        # Check data quality
        data_quality = "✅ Good" if len(analysis_df) >= 5 else "⚠️ Limited" if len(analysis_df) >= 3 else "❌ Poor"
        st.metric("Data Points", data_quality, f"{len(analysis_df)}")
    
    # Interpretation based on m value and R²
    st.subheader("📊 Interpretation")
    
    # Scenario 1: Negative m value (Important Warning!)
    if m_est < 0:
        st.error("""
        ⚠️ **CRITICAL: Negative Gas Cap Ratio (m) Detected!**
        
        **Possible Causes:**
        1. **Water Influx**: Strong aquifer support reducing apparent gas cap size
        2. **No Gas Cap**: Reservoir likely has no initial gas cap
        3. **Data Issues**: Check PVT data quality, especially Bg values
        4. **Reservoir Communication**: Leaks or communication with other zones
        5. **Incorrect m from volumetric**: Volumetric m overestimated
        
        **Recommended Actions:**
        - Re-examine reservoir geology for aquifer support
        - Check if m from volumetric maps is reliable
        - Consider water influx analysis (F/Eo vs ΣΔP*Eo plot)
        - Verify PVT data, especially Bg conversion (cu ft/SCF → bbl/SCF)
        """)
    
    # Scenario 2: Very small or zero m value
    elif m_est < 0.01:
        st.info("""
        ✅ **Depletion Drive Dominant**
        
        **Characteristics:**
        - Minimal or no gas cap (m ≈ 0)
        - Primary drive: Solution gas drive
        - Reservoir likely undersaturated or with very small gas cap
        - Pressure decline typically rapid
        - Recovery factor typically 5-30%
        
        **Verification:**
        - Check F vs Eo plot for straight line through origin
        - If straight line: Confirms depletion drive
        """)
    
    # Scenario 3: Moderate gas cap
    elif m_est < 0.3:
        st.success(f"""
        ✅ **Gas Cap Drive Present (m = {m_est:.3f})**
        
        **Characteristics:**
        - Moderate gas cap expansion drive
        - Combined solution gas + gas cap drive
        - Better pressure maintenance than depletion drive
        - Recovery factor typically 20-40%
        
        **Verification:**
        - Good linear fit (R² = {r2:.3f}) confirms analysis
        - Gas cap provides energy for oil recovery
        """)
    
    # Scenario 4: Large gas cap
    else:
        st.warning(f"""
        ⚠️ **Large Gas Cap Detected (m = {m_est:.3f})**
        
        **Characteristics:**
        - Very large gas cap relative to oil zone
        - Strong gas cap expansion drive
        - Risk of gas coning/cusping
        - May need gas injection control
        
        **Concerns:**
        - Check if m value realistic from geology
        - Verify volumetric m estimate
        - Consider gas cap blowdown strategy
        """)
    
    # Scenario 5: Poor R²
    if r2 < 0.85:
        st.warning("""
        ⚠️ **Poor Linear Fit Detected (R² < 0.85)**
        
        **Possible Causes:**
        1. **Water Influx**: Strong aquifer support not accounted for
        2. **Multiple Drive Mechanisms**: Combination of drives
        3. **Poor Data Quality**: Inconsistent pressure or production data
        4. **Pressure Averaging Issues**: Reservoir pressure not representative
        5. **PVT Data Issues**: Incorrect Bo, Rs, or Bg values
        
        **Recommended Actions:**
        - Try F vs Eo plot (depletion drive check)
        - Consider water influx analysis
        - Review data quality and consistency
        - Check pressure survey methodology
        """)
    
    # Scenario 6: Excellent R²
    elif r2 > 0.95:
        st.success("""
        ✅ **Excellent Linear Fit (R² > 0.95)**
        
        **Interpretation:**
        - Strong confirmation of gas cap drive mechanism
        - Good data quality and consistent reservoir behavior
        - Reliable N and m estimates
        - Suitable for performance prediction
        
        **Confidence Level: High**
        """)
    
    # Add diagnostic plots for other scenarios
    st.subheader("🔧 Additional Diagnostic Checks")
    
    # Check for water influx (F/Eo vs Cumulative ΔP*Eo)
    col1, col2 = st.columns(2)
    
    with col1:
        # F vs Eo plot (depletion drive check)
        valid_f_eo = calc_df[calc_df['Eo'] > 0].dropna(subset=['F', 'Eo'])
        if len(valid_f_eo) >= 2:
            fig_depletion = go.Figure()
            fig_depletion.add_trace(go.Scatter(
                x=valid_f_eo['Eo'].values,
                y=valid_f_eo['F'].values,
                mode='markers',
                name='Data Points'
            ))
            
            # Fit line through origin for depletion drive
            if len(valid_f_eo) >= 1:
                N_depletion = np.sum(valid_f_eo['F'] * valid_f_eo['Eo']) / np.sum(valid_f_eo['Eo']**2)
                x_line_dep = np.linspace(0, max(valid_f_eo['Eo']) * 1.1, 100)
                y_line_dep = N_depletion * x_line_dep
                fig_depletion.add_trace(go.Scatter(
                    x=x_line_dep,
                    y=y_line_dep,
                    mode='lines',
                    name=f'Depletion: F = {N_depletion:.1f} × Eo',
                    line=dict(color='green', width=2)
                ))
            
            fig_depletion.update_layout(
                title='F vs Eo (Depletion Drive Check)',
                xaxis_title='Eo (bbl/STB)',
                yaxis_title='F (MMbbl)',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_depletion, use_container_width=True)
            
            # Depletion drive interpretation
            if r2 < 0.85 and m_est < 0.01:
                st.info("""
                **Depletion Drive Likely:**
                - F vs Eo shows good linearity through origin
                - m ≈ 0 confirms no significant gas cap
                - Consider solution gas drive as primary mechanism
                """)
    
    with col2:
        # Water influx indicator: F/Eo vs Cumulative ΔP
        calc_df['Cum_ΔP_Eo'] = calc_df['ΔP'] * calc_df['Eo']
        valid_water = calc_df[calc_df['Eo'] > 0].dropna(subset=['F/Eo', 'Cum_ΔP_Eo'])
        
        if len(valid_water) >= 2:
            fig_water = go.Figure()
            fig_water.add_trace(go.Scatter(
                x=valid_water['Cum_ΔP_Eo'].values,
                y=valid_water['F/Eo'].values,
                mode='markers',
                name='Data Points'
            ))
            
            fig_water.update_layout(
                title='F/Eo vs ΔP×Eo (Water Influx Check)',
                xaxis_title='ΔP × Eo',
                yaxis_title='F/Eo',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_water, use_container_width=True)
            
            # Water influx interpretation
            if m_est < 0:
                st.warning("""
                **Water Influx Possible:**
                - Negative m suggests aquifer support
                - Check F/Eo vs ΔP×Eo for upward trend
                - Consider Carter-Tracy or Fetkovich aquifer model
                """)
else:
    st.error("❌ Not enough valid data points for analysis")

# ============================================================
# SCENARIO-BASED ANALYSIS
# ============================================================

st.header("📈 Scenario-Based Analysis")

# Create tabs for different scenarios
tab1, tab2, tab3, tab4 = st.tabs([
    "Scenario 1: Depletion Drive",
    "Scenario 2: Gas Cap Drive",
    "Scenario 3: Gas Cap + Compressibility",
    "Scenario 4: Water Influx"
])

# ============================================================
# SCENARIO 1: Depletion Drive (No Gas Cap, No Water)
# ============================================================
with tab1:
    st.subheader("Scenario 1: Depletion Drive (F = N × Eo)")
    
    # Equation: F = N × Eo
    valid_data = analysis_df.dropna(subset=['F', 'Eo'])
    valid_data = valid_data[np.isfinite(valid_data['F']) & np.isfinite(valid_data['Eo'])]
    
    if len(valid_data) >= 2:
        x = valid_data['Eo'].values
        y = valid_data['F'].values
        
        # Linear regression through origin
        slope = np.sum(x * y) / np.sum(x**2)
        N_estimated = slope  # MMSTB
        
        # Calculate R-squared
        y_pred = slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated N (OOIP)", f"{N_estimated:.2f} MMSTB")
        with col2:
            st.metric("R² Value", f"{r_squared:.4f}")
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
        
        x_line = np.linspace(0, max(x)*1.1, 100)
        y_line = slope * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', 
                                name=f'F = {slope:.2f}×Eo',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(title='Depletion Drive: F vs Eo',
                         xaxis_title='Eo (bbl/STB)',
                         yaxis_title='F (million bbl)',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        if r_squared > 0.95:
            st.success("✅ Excellent fit - Depletion drive likely")
        elif r_squared > 0.85:
            st.info("📈 Good fit - Depletion drive probable")
        else:
            st.warning("⚠️ Poor fit - Other mechanisms may be present")
    else:
        st.error("❌ Not enough valid data points")

# ============================================================
# SCENARIO 2: Gas Cap Drive (Unknown m)
# ============================================================
with tab2:
    st.subheader("Scenario 2: Gas Cap Drive (F/Eo = N + mN × Eg/Eo)")
    
    # Equation: F/Eo = N + mN × (Eg/Eo)
    valid_data = analysis_df.dropna(subset=['F/Eo', 'Eg/Eo'])
    valid_data = valid_data[np.isfinite(valid_data['F/Eo']) & np.isfinite(valid_data['Eg/Eo'])]
    
    if len(valid_data) >= 2:
        x = valid_data['Eg/Eo'].values
        y = valid_data['F/Eo'].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        
        # Calculate N and m
        N_estimated = intercept  # MMSTB
        if intercept > 0:
            m_estimated = slope / intercept
        else:
            m_estimated = 0
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated N", f"{N_estimated:.2f} MMSTB")
        with col2:
            st.metric("Gas-Cap Ratio (m)", f"{m_estimated:.4f}")
        with col3:
            st.metric("R² Value", f"{r_squared:.4f}")
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
        
        x_line = np.linspace(min(x)*0.9, max(x)*1.1, 100)
        y_line = slope * x_line + intercept
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                name=f'y = {slope:.2f}x + {intercept:.2f}',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(title='Gas Cap Drive: F/Eo vs Eg/Eo',
                         xaxis_title='Eg/Eo',
                         yaxis_title='F/Eo',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        if m_estimated < 0.01:
            st.info("🔍 Minimal gas cap (m ≈ 0)")
        elif m_estimated < 0.3:
            st.success(f"🔍 Moderate gas cap (m = {m_estimated:.3f})")
        else:
            st.warning(f"🔍 Large gas cap (m = {m_estimated:.3f})")
    else:
        st.error("❌ Not enough valid data points")

# ============================================================
# SCENARIO 3: Gas Cap + Compressibility
# ============================================================
with tab3:
    st.subheader("Scenario 3: Gas Cap + Compressibility")
    
    # User input for m
    m_input = st.number_input("Enter gas-cap ratio (m):", 
                             value=0.1, min_value=0.0, max_value=2.0, step=0.01,
                             key="tab3_m")
    
    # Equation: F = N × (Eo + mEg + Efw)
    analysis_df['Eo+mEg+Efw'] = analysis_df['Eo'] + m_input * analysis_df['Eg'] + analysis_df['Efw']
    
    valid_data = analysis_df.dropna(subset=['F', 'Eo+mEg+Efw'])
    valid_data = valid_data[np.isfinite(valid_data['F']) & np.isfinite(valid_data['Eo+mEg+Efw'])]
    
    if len(valid_data) >= 2:
        x = valid_data['Eo+mEg+Efw'].values
        y = valid_data['F'].values
        
        # Linear regression through origin
        slope = np.sum(x * y) / np.sum(x**2)
        N_estimated = slope
        
        # R-squared
        y_pred = slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Display
        st.metric("Estimated N", f"{N_estimated:.2f} MMSTB")
        st.metric("R² Value", f"{r_squared:.4f}")
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data Points'))
        
        x_line = np.linspace(0, max(x)*1.1, 100)
        y_line = slope * x_line
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                                name=f'F = {slope:.2f}×(Eo+{m_input}Eg+Efw)',
                                line=dict(color='red', dash='dash')))
        
        fig.update_layout(title=f'Gas Cap + Compressibility (m={m_input})',
                         xaxis_title='Eo + mEg + Efw (bbl/STB)',
                         yaxis_title='F (million bbl)',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SCENARIO 4: Water Influx
# ============================================================
with tab4:
    st.subheader("Scenario 4: Water Influx Analysis")
    
    st.info("""
    **Water influx requires iterative aquifer modeling.**
    This diagnostic plot helps identify if water influx might be present.
    """)
    
    # Plot F/Eo vs something to check for water influx pattern
    valid_data = analysis_df.dropna(subset=['F/Eo', 'Eg/Eo'])
    valid_data = valid_data[np.isfinite(valid_data['F/Eo']) & np.isfinite(valid_data['Eg/Eo'])]
    
    if len(valid_data) >= 2:
        x = valid_data['Eg/Eo'].values
        y = valid_data['F/Eo'].values
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', 
                                name='F/Eo vs Eg/Eo',
                                line=dict(color='blue')))
        
        fig.update_layout(title='Water Influx Diagnostic',
                         xaxis_title='Eg/Eo',
                         yaxis_title='F/Eo',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **Straight line**: No water influx
        - **Upward curvature**: Possible water influx
        - **Scattered points**: Multiple drive mechanisms
        """)

# ============================================================
# SUMMARY TABLE
# ============================================================

st.header("📊 Summary of All Scenarios")

# Collect results from all scenarios
summary_data = []

# Scenario 1 results
valid_s1 = analysis_df.dropna(subset=['F', 'Eo'])
if len(valid_s1) >= 2:
    x = valid_s1['Eo'].values
    y = valid_s1['F'].values
    slope = np.sum(x * y) / np.sum(x**2)
    y_pred = slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    summary_data.append({
        'Scenario': 'Depletion Drive',
        'Equation': 'F = N × Eo',
        'N (MMSTB)': f"{slope:.2f}",
        'm': '0.0000',
        'R²': f"{r_squared:.4f}"
    })

# Scenario 2 results
valid_s2 = analysis_df.dropna(subset=['F/Eo', 'Eg/Eo'])
if len(valid_s2) >= 2:
    x = valid_s2['Eg/Eo'].values
    y = valid_s2['F/Eo'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    m_est = slope / intercept if intercept > 0 else 0
    summary_data.append({
        'Scenario': 'Gas Cap Drive',
        'Equation': 'F/Eo = N + mN × Eg/Eo',
        'N (MMSTB)': f"{intercept:.2f}",
        'm': f"{m_est:.4f}",
        'R²': f"{r_squared:.4f}"
    })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================

st.markdown("---")
st.subheader("📋 Summary & Recommendations")

if len(analysis_df) >= 2:
    # Create summary table
    summary_data = {
        'Parameter': ['Original Oil in Place (N)', 'Gas Cap Ratio (m)', 
                     'Data Quality (R²)', 
                     'Drive Mechanism', 'Confidence Level'],
        'Value': [
            f"{N_est:.2f} MMSTB",
            f"{m_est:.4f}",
            f"{r2:.4f} ({'Excellent' if r2 > 0.95 else 'Good' if r2 > 0.85 else 'Poor'})",
            f"{'Gas Cap Drive' if m_est > 0.01 else 'Depletion Drive'}",
            f"{'High' if r2 > 0.95 and len(analysis_df) >= 5 else 'Medium' if r2 > 0.85 else 'Low'}"
        ],
        'Interpretation': [
            f"{'✓ Realistic' if N_est > 0 else '✗ Check data'}",
            f"{'✓ Positive gas cap' if m_est > 0.01 else '✗ Negative/small - check water influx' if m_est < 0 else '✓ Minimal gas cap'}",
            f"{'✓ Good fit' if r2 > 0.85 else '✗ Poor fit - check data'}",
            f"{'✓ Identified' if r2 > 0.85 else '✗ Unclear - multiple drives possible'}",
            f"{'✓ Reliable' if r2 > 0.95 else '⚠️ Use with caution' if r2 > 0.85 else '✗ Low reliability'}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
    
    # Recommendations
    st.subheader("🎯 Recommended Actions")
    
    if m_est < 0:
        st.markdown("""
        1. **Immediate Priority**: Investigate negative m value
           - Review geological model for aquifer support
           - Verify volumetric m calculation
           - Check for data errors in production/PVT
           
        2. **Consider Water Influx Analysis**
           - Prepare F/Eo vs ΣΔP*Eo plot
           - Apply Carter-Tracy or Fetkovich aquifer model
           - Estimate aquifer size and strength
           
        3. **Data Verification**
           - Cross-check all PVT data conversions
           - Verify pressure data quality
           - Review production allocation
        """)
    
    elif r2 < 0.85:
        st.markdown("""
        1. **Data Quality Improvement**
           - Review pressure measurement methodology
           - Check PVT data consistency
           - Verify production data allocation
           
        2. **Alternative Analysis Methods**
           - Try volumetric balance method
           - Consider decline curve analysis
           - Use simulation for history matching
           
        3. **Additional Data Collection**
           - More frequent pressure surveys
           - Updated PVT analysis
           - Detailed production logging
        """)
    
    else:
        st.markdown("""
        1. **Confirmation Analysis**
           - Compare with volumetric estimates
           - Check consistency with production history
           - Validate with decline curve analysis
           
        2. **Reservoir Management**
           - Optimize gas cap management strategy
           - Plan for pressure maintenance
           - Consider gas injection if needed
           
        3. **Performance Prediction**
           - Use established model for forecasts
           - Plan development drilling
           - Optimize recovery strategy
        """)

# ============================================================
# ADDITIONAL DIAGNOSTIC PLOTS
# ============================================================

st.header("📉 Pressure & Production Analysis")

col1, col2 = st.columns(2)

with col1:
    # Plot F vs Eo
    valid_f_eo = calc_df[calc_df['Eo'] > 0].dropna(subset=['F', 'Eo'])
    if len(valid_f_eo) >= 2:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=valid_f_eo['Eo'].values,
            y=valid_f_eo['F'].values,
            mode='markers',
            name='F vs Eo'
        ))
        fig1.update_layout(
            title='F vs Eo (Depletion Drive Check)',
            xaxis_title='Eo (bbl/STB)',
            yaxis_title='F (MMbbl)',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Plot pressure decline
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=calc_df['Np'].values,
        y=calc_df['P'].values,
        mode='lines+markers',
        name='Pressure Decline',
        line=dict(color='red', width=2)
    ))
    
    # Add pressure decline rate
    if len(calc_df) >= 2:
        pressure_decline = (calc_df['P'].iloc[0] - calc_df['P'].iloc[-1]) / (calc_df['Np'].iloc[-1] - calc_df['Np'].iloc[0])
        fig2.add_annotation(
            x=0.95, y=0.95,
            xref="paper", yref="paper",
            text=f"Decline: {pressure_decline:.1f} psi/MMSTB",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    
    fig2.update_layout(
        title='Pressure Decline Curve',
        xaxis_title='Np (MMSTB)',
        yaxis_title='Pressure (psi)',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown("---")
st.subheader("📥 Download Results")

if len(calc_df) > 0:
    # Prepare CSV with all calculated values
    csv = calc_df.to_csv(index=False)
    st.download_button(
        label="Download Calculation Results (CSV)",
        data=csv,
        file_name="havlena_odeh_results.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Prepare summary report
    if len(analysis_df) >= 2:
        report_text = f"""HAVLENA-ODEH ANALYSIS REPORT
==============================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Points: {len(calc_df)} valid points, {len(analysis_df)} for analysis

RESULTS:
--------
Original Oil in Place (N): {N_est:.2f} MMSTB
Gas Cap Ratio (m): {m_est:.4f}
Regression R²: {r2:.4f}

INTERPRETATION:
---------------
Drive Mechanism: {'Gas Cap Drive' if m_est > 0.01 else 'Depletion Drive'}
Linear Fit Quality: {'Excellent' if r2 > 0.95 else 'Good' if r2 > 0.85 else 'Poor'}
Confidence Level: {'High' if r2 > 0.95 and len(analysis_df) >= 5 else 'Medium' if r2 > 0.85 else 'Low'}

RECOMMENDATIONS:
---------------
"""
        
        if m_est < 0:
            report_text += "- INVESTIGATE NEGATIVE m VALUE: Check for water influx or data errors\n"
        elif r2 < 0.85:
            report_text += "- IMPROVE DATA QUALITY: Review pressure and PVT measurements\n"
        else:
            report_text += "- PROCEED WITH CONFIDENCE: Results appear reliable for planning\n"
        
        st.download_button(
            label="Download Analysis Report (TXT)",
            data=report_text,
            file_name="havlena_odeh_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# ============================================================
# FORMULAS REFERENCE
# ============================================================

with st.expander("📚 Formulas & Methodology Reference"):
    st.markdown("""
    ### **Havlena-Odeh Formulas:**
    
    **Producing GOR:**
    ```
    Rp = Gp / Np  [SCF/STB]
    (Gp in MMSCF, Np in MMSTB)
    ```
    
    **Underground Withdrawal:**
    ```
    F = Np[Bo + (Rp - Rs)Bg] + WpBw  [MMbbl]
    ```
    
    **Oil Expansion:**
    ```
    Eo = (Bo - Boi) + (Rsi - Rs)Bg  [bbl/STB]
    ```
    
    **Gas Cap Expansion:**
    ```
    Eg = Boi × (Bg/Bgi - 1)  [bbl/STB]
    ```
    
    **Straight Line Equation:**
    ```
    F/Eo = N + mN × (Eg/Eo)
    ```
    - Intercept = N (OOIP in MMSTB)
    - Slope = m × N
    - m = Slope / Intercept

    ### **Drive Mechanism Interpretation:**

    **1. Negative m value (m < 0):**
    - Strong indication of water influx
    - Possible volumetric m overestimate
    - Check for aquifer support

    **2. Small m value (0 < m < 0.01):**
    - Depletion/solution gas drive dominant
    - Minimal gas cap expansion
    - F vs Eo should be linear through origin

    **3. Moderate m value (0.01 < m < 0.3):**
    - Combined gas cap + solution gas drive
    - Good pressure maintenance
    - Typical for many reservoirs

    **4. Large m value (m > 0.3):**
    - Large gas cap relative to oil zone
    - Strong gas expansion drive
    - Check volumetric consistency

    **5. Poor R² (< 0.85):**
    - Multiple drive mechanisms
    - Water influx not accounted for
    - Data quality issues
    - Need for more complex analysis
    """)

# ============================================================
# SIDEBAR INFO
# ============================================================

with st.sidebar:
    st.markdown("---")
    st.header("ℹ️ Analysis Guide")
    
    st.markdown("""
    **Interpretation Guide:**
    
    ✅ **Good Results:**
    - R² > 0.95, m > 0
    - Clear gas cap drive
    - Reliable N estimate
    
    ⚠️ **Caution Needed:**
    - R² < 0.85
    - Multiple drives possible
    - Check data quality
    
    ❌ **Problematic:**
    - m < 0 (negative gas cap)
    - Water influx likely
    - Re-examine assumptions
    
    **Quick Checks:**
    1. F vs Eo: Straight line = depletion drive
    2. F/Eo vs Eg/Eo: Linear = gas cap drive
    3. Negative slope: Water influx suspected
    """)