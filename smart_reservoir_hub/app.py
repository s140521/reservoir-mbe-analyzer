import streamlit as st

st.set_page_config(
    page_title="Reservoir MBE-Pro Analyzer",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        color: #0f172a; 
        text-align: center; 
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header { 
        font-size: 1.2rem; 
        color: #0369a1; 
        text-align: center;
        margin-bottom: 1rem;
    }
    .quote {
        font-size: 0.95rem;
        color: #0369a1;  
        text-align: center;
        line-height: 1.6;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .quote-citation {
        font-size: 0.8rem;  /* Smaller than the quote */
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛢️ Reservoir MBE-Pro Analyzer</div>', unsafe_allow_html=True)

# Quote in separate lines with baby blue color
st.markdown("""
<div class="quote">
    Every reservoir has its own personality;<br>
    our equations are merely translators.<br>
    <div class="quote-citation">— L.P. Dake, Fundamentals of Reservoir Engineering</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📋 **About This Dashboard**
    
    This dashboard provides a complete suite of reservoir engineering tools:
    
    - **Material Balance Analysis** (Havlena-Odeh method)
    - **Drive Mechanism Analysis**
    - **Aquifer Influence Assessment**
    - **p/Z Diagnostics**
    - **Production Forecasting**
    
    Designed for **PNGE4412 - Reservoir Engineering, Fall 2025**.
    """)

with col2:
    st.markdown("""
    ### 🚀 **How to Use**
    
    1. **Start with "Upload Data"** to load your production and PVT data
    2. **Proceed through pages in order** for comprehensive analysis
    3. **Download results** from each analysis page
    4. **Use sample data** for testing if needed
    """)

st.markdown("---")

# Footer with your names
st.markdown("""
<div class="footer">
    <strong>Designed by:</strong> Maithaa Said, Safaa Said, Al Mazyoon Khalil
</div>
""", unsafe_allow_html=True)