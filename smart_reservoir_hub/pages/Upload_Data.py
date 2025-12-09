import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Upload Data | Reservoir Analysis", layout="wide")

st.title("📂 Upload Production & PVT Data")

st.markdown("""
### **Format Requirements:**
Upload an Excel file with **two sheets**:

1. **Production Data Sheet**: Columns: P, Rs, Bo, Bg, Np, Gp
2. **Initial Parameters Sheet**: Columns: pi, Boi, Bgi, Rsi, Swc, Cw, Cf
3. **Make sure Np & Gp in MM and Bg in bbl/SCF**

*Column names can vary slightly - the system will auto-detect them.*
""")

# Initialize session state keys
if "production_data" not in st.session_state:
    st.session_state["production_data"] = None
if "production_data_std" not in st.session_state:
    st.session_state["production_data_std"] = None
if "initial_params_raw" not in st.session_state:
    st.session_state["initial_params_raw"] = None
if "initial_params" not in st.session_state:
    st.session_state["initial_params"] = None
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"], 
                                 help="Upload Excel file with production data and initial parameters")

# Process uploaded file
if uploaded_file is not None:
    try:
        # Read Excel file
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.success(f"✅ File loaded successfully! Sheets found: {sheet_names}")
        
        # Auto-detect sheets
        prod_sheet = None
        init_sheet = None
        
        # Look for production data sheet
        for sheet in sheet_names:
            df_test = pd.read_excel(xls, sheet_name=sheet, nrows=3)
            cols = [str(c).lower().strip() for c in df_test.columns]
            
            prod_count = 0
            for indicator in ['np', 'gp', 'bo', 'bg', 'rs']:
                if any(indicator in col for col in cols):
                    prod_count += 1
            
            if prod_count >= 3:
                prod_sheet = sheet
                break
        
        # Look for initial parameters sheet
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            if any(keyword in sheet_lower for keyword in ['initial', 'init', 'param', 'property', 'config', 'setup']):
                init_sheet = sheet
                break
        
        # Fallback logic
        if prod_sheet is None and len(sheet_names) >= 1:
            prod_sheet = sheet_names[0]
            st.info(f"Using first sheet '{prod_sheet}' as production data")
        
        if init_sheet is None and len(sheet_names) >= 2:
            init_sheet = sheet_names[1]
            st.info(f"Using second sheet '{init_sheet}' for initial parameters")
        elif init_sheet is None and len(sheet_names) == 1:
            init_sheet = prod_sheet
            st.warning("Only one sheet found. Will attempt to extract parameters from same sheet.")
        
        # Load production data
        prod_df = pd.read_excel(xls, sheet_name=prod_sheet)
        st.session_state["production_data"] = prod_df
        
        # ============================================================
        # COLUMN STANDARDIZATION
        # ============================================================
        col_mapping = {}
        used_names = set()
        
        for col in prod_df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            new_name = None
            
            # Specific pattern matching
            if any(term in col_lower for term in ['np', 'n_p', 'cumulative oil', 'oil produced']):
                new_name = 'Np'
            elif any(term in col_lower for term in ['gp', 'g_p', 'cumulative gas', 'gas produced']):
                new_name = 'Gp'
            elif any(term in col_lower for term in ['bo', 'oil fvf', 'oil formation']):
                new_name = 'Bo'
            elif any(term in col_lower for term in ['bg', 'gas fvf', 'gas formation']):
                new_name = 'Bg'
            elif any(term in col_lower for term in ['rs', 'r_s', 'solution gas', 'gor', 'gas oil ratio']):
                new_name = 'Rs'
            elif any(term in col_lower for term in ['pressure', 'p_res', 'p_avg', 'reservoir pressure']):
                new_name = 'P'
            elif col_lower == 'p':
                new_name = 'P'
            else:
                new_name = col_str
            
            # Handle duplicates
            if new_name in used_names:
                suffix = 1
                while f"{new_name}_{suffix}" in used_names:
                    suffix += 1
                new_name = f"{new_name}_{suffix}"
            
            col_mapping[col] = new_name
            used_names.add(new_name)
        
        prod_df_std = prod_df.rename(columns=col_mapping)
        
        if prod_df_std.columns.duplicated().any():
            prod_df_std.columns = [f"{col}_{i}" if col in prod_df_std.columns[:i] else col 
                                  for i, col in enumerate(prod_df_std.columns)]
        
        st.session_state["production_data_std"] = prod_df_std
        
        # ============================================================
        # LOAD INITIAL PARAMETERS
        # ============================================================
        std_params = {}
        
        if init_sheet == prod_sheet:
            # Extract from first row of production data (initial conditions)
            first_row = prod_df.iloc[0]
            
            # Extract initial conditions from first data point
            extracted_params = {}
            if 'P' in prod_df_std.columns:
                extracted_params['pi'] = float(first_row['P'])
            if 'Bo' in prod_df_std.columns:
                extracted_params['Boi'] = float(first_row['Bo'])
            if 'Bg' in prod_df_std.columns:
                extracted_params['Bgi'] = float(first_row['Bg'])
            if 'Rs' in prod_df_std.columns:
                extracted_params['Rsi'] = float(first_row['Rs'])
            
            std_params.update(extracted_params)
            
            st.session_state["initial_params_raw"] = pd.DataFrame({
                'Property': list(extracted_params.keys()),
                'Value': list(extracted_params.values())
            })
            
            if extracted_params:
                st.info(f"Extracted {len(extracted_params)} initial conditions from first production data point")
        else:
            # Load from separate sheet
            init_df = pd.read_excel(xls, sheet_name=init_sheet)
            st.session_state["initial_params_raw"] = init_df
            
            # Parse parameters from sheet
            param_dict = {}
            
            if 'Property' in init_df.columns and 'Value' in init_df.columns:
                for _, row in init_df.iterrows():
                    key = str(row['Property']).strip()
                    value = row['Value']
                    if pd.notna(key) and pd.notna(value):
                        param_dict[key] = value
            elif len(init_df.columns) >= 2:
                for _, row in init_df.iterrows():
                    if pd.notna(row.iloc[0]) and pd.notna(row.iloc[1]):
                        key = str(row.iloc[0]).strip()
                        value = row.iloc[1]
                        param_dict[key] = value
            
            # Standardize parameter names
            for key, value in param_dict.items():
                if pd.isna(value):
                    continue
                    
                key_lower = str(key).lower().strip()
                try:
                    if isinstance(value, (int, float, np.number)):
                        value = float(value)
                    else:
                        value = float(str(value).replace(',', ''))
                except:
                    continue
                
                if any(term in key_lower for term in ['initial', 'pi', 'p_i', 'initial pressure', 'p initial']):
                    std_params['pi'] = value
                elif any(term in key_lower for term in ['boi', 'b_oi', 'initial bo', 'bo initial']):
                    std_params['Boi'] = value
                elif any(term in key_lower for term in ['bgi', 'b_gi', 'initial bg', 'bg initial']):
                    std_params['Bgi'] = value
                elif any(term in key_lower for term in ['rsi', 'r_si', 'initial rs', 'rs initial']):
                    std_params['Rsi'] = value
                elif any(term in key_lower for term in ['swc', 's_wc', 'connate', 'water saturation']):
                    std_params['Swc'] = value
                elif any(term in key_lower for term in ['cw', 'c_w', 'water comp', 'water compressibility']):
                    std_params['Cw'] = value
                elif any(term in key_lower for term in ['cf', 'c_f', 'formation comp', 'rock comp', 'formation compressibility']):
                    std_params['Cf'] = value
                elif any(term in key_lower for term in ['porosity', 'phi']):
                    std_params['phi'] = value
        
        # Store in session state
        st.session_state["initial_params"] = std_params
        st.session_state["data_loaded"] = True
        
        st.success("✅ Data loaded successfully!")
        
        # ============================================================
        # DISPLAY PREVIEW
        # ============================================================
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Production Data Preview")
            st.dataframe(prod_df_std.head(), use_container_width=True)
            st.write(f"**Shape:** {prod_df_std.shape[0]} rows × {prod_df_std.shape[1]} columns")
        
        with col2:
            st.subheader("⚙️ Parameters Loaded")
            if std_params:
                params_df = pd.DataFrame({
                    'Parameter': list(std_params.keys()),
                    'Value': list(std_params.values())
                })
                st.dataframe(params_df, use_container_width=True)
            else:
                st.warning("No parameters loaded from file")
            
            # Check for required production columns
            required_cols = ['P', 'Np', 'Gp', 'Bo', 'Bg', 'Rs']
            missing_cols = [col for col in required_cols if col not in prod_df_std.columns]
            
            if missing_cols:
                st.error(f"⚠️ Missing required columns: {missing_cols}")
                st.info("Analysis may not work correctly. Please check your data format.")
            else:
                st.success("✓ All required production columns present")
                
                # Check for critical initial parameters
                critical_params = ['pi', 'Boi', 'Bgi', 'Rsi']
                missing_critical = [p for p in critical_params if p not in std_params]
                if missing_critical:
                    st.warning(f"⚠️ Missing critical parameters: {missing_critical}")
                    st.info("These parameters are required for Havlena-Odeh analysis.")
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("Please ensure your Excel file has the correct format and no empty rows/columns.")

# ============================================================
# NAVIGATION AND NEXT STEPS
# ============================================================
st.markdown("---")

col1, col2 = st.columns([3, 1])

with col1:
    st.info("""
    💡 **Next Step:** 
    1. Go to the **Havlena-Odeh** page to perform material balance analysis
    2. Or visit other analysis pages for different reservoir engineering calculations
    """)

with col2:
    if st.session_state["data_loaded"]:
        st.success("✅ Data Ready for Analysis")
    else:
        st.warning("⏳ Waiting for Data")

# Sidebar information
with st.sidebar:
    st.header("📋 Data Requirements")
    
    st.markdown("""
    ### Required Production Columns:
    - **P**: Reservoir pressure (psia)
    - **Np**: Cumulative oil production (STB)
    - **Gp**: Cumulative gas production (SCF)
    - **Bo**: Oil formation volume factor (rb/STB)
    - **Bg**: Gas formation volume factor (rb/SCF)
    - **Rs**: Solution gas-oil ratio (SCF/STB)
    
    ### Required Initial Parameters:
    - **pi**: Initial reservoir pressure
    - **Boi**: Initial oil FVF
    - **Bgi**: Initial gas FVF
    - **Rsi**: Initial solution GOR
    - **Swc**: Connate water saturation
    - **Cw**: Water compressibility
    - **Cf**: Formation compressibility
    """)
    
    st.markdown("---")
    st.caption("**PNGE4412 - Reservoir Engineering**")
 