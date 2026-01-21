"""
Wine Cultivar Origin Prediction System
Web Application using Streamlit

Student: Victor Emeka
Matric: 23cg034065
Algorithm: Random Forest Classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Wine Cultivar Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #FF5252;
        border: none;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        # Try multiple path strategies
        base_dir = Path(__file__).parent
        
        # Strategy 1: Relative to app.py
        model_path = base_dir / 'model' / 'wine_cultivar_model.pkl'
        scaler_path = base_dir / 'model' / 'scaler.pkl'
        
        # Strategy 2: If not found, try current working directory
        if not model_path.exists():
            model_path = Path('model/wine_cultivar_model.pkl')
            scaler_path = Path('model/scaler.pkl')
        
        # Debug info
        st.sidebar.info(f"📂 Loading from: {model_path.parent.absolute()}")
        
        if not model_path.exists():
            st.error(f"❌ Model file not found at: {model_path.absolute()}")
            st.error(f"📂 Current directory: {Path.cwd()}")
            st.error(f"📂 Script directory: {base_dir.absolute()}")
            st.error(f"📁 Directory contents: {list(base_dir.glob('*'))}")
            if (base_dir / 'model').exists():
                st.error(f"📁 Model directory contents: {list((base_dir / 'model').glob('*'))}")
            return None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        st.sidebar.success("✅ Model loaded successfully!")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

# Feature information
FEATURE_INFO = {
    'alcohol': {
        'name': 'Alcohol Content',
        'range': (11.0, 15.0),
        'unit': '%',
        'description': 'Alcohol percentage in wine'
    },
    'malic_acid': {
        'name': 'Malic Acid',
        'range': (0.5, 6.0),
        'unit': 'g/L',
        'description': 'Malic acid concentration'
    },
    'ash': {
        'name': 'Ash Content',
        'range': (1.0, 4.0),
        'unit': 'g/L',
        'description': 'Ash content in wine'
    },
    'total_phenols': {
        'name': 'Total Phenols',
        'range': (0.5, 4.0),
        'unit': 'mg/L',
        'description': 'Total phenolic compounds'
    },
    'flavanoids': {
        'name': 'Flavanoids',
        'range': (0.0, 6.0),
        'unit': 'mg/L',
        'description': 'Flavanoid content'
    },
    'color_intensity': {
        'name': 'Color Intensity',
        'range': (1.0, 13.0),
        'unit': 'units',
        'description': 'Wine color intensity'
    }
}

CULTIVAR_INFO = {
    0: {
        'name': 'Cultivar 0',
        'description': 'First wine cultivar with distinct chemical characteristics',
        'emoji': '🍇'
    },
    1: {
        'name': 'Cultivar 1',
        'description': 'Second wine cultivar with unique properties',
        'emoji': '🍷'
    },
    2: {
        'name': 'Cultivar 2',
        'description': 'Third wine cultivar with specific chemical profile',
        'emoji': '🍾'
    }
}

def main():
    # Header
    st.markdown("<h1>🍷 Wine Cultivar Origin Prediction System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>📊 About this Application</strong><br>
        This machine learning application predicts wine cultivar (origin/class) based on chemical properties 
        using a <strong>Random Forest Classifier</strong>. Enter the chemical properties of your wine sample below.
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.error("⚠️ Failed to load model. Please ensure model files exist in the 'model' directory.")
        return
    
    # Sidebar - Information
    with st.sidebar:
        st.header("📋 Application Info")
        st.markdown("---")
        st.markdown("""
        **Student:** Victor Emeka  
        **Matric:** 23cg034065  
        **Algorithm:** Random Forest Classifier  
        **Model Persistence:** Joblib
        """)
        
        st.markdown("---")
        st.header("🎯 Features Used")
        st.markdown("""
        1. Alcohol Content
        2. Malic Acid
        3. Ash Content
        4. Total Phenols
        5. Flavanoids
        6. Color Intensity
        """)
        
        st.markdown("---")
        st.header("🍇 Cultivar Classes")
        for cultivar_id, info in CULTIVAR_INFO.items():
            st.markdown(f"{info['emoji']} **{info['name']}**")
        
        st.markdown("---")
        st.markdown("### 🔬 Sample Data")
        if st.button("Load Sample Wine"):
            st.session_state.load_sample = True
    
    # Main content - Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔬 Enter Wine Chemical Properties")
        
        # Check if sample should be loaded
        if 'load_sample' in st.session_state and st.session_state.load_sample:
            sample_values = {
                'alcohol': 13.2,
                'malic_acid': 2.0,
                'ash': 2.3,
                'total_phenols': 2.8,
                'flavanoids': 3.0,
                'color_intensity': 5.5
            }
            st.session_state.load_sample = False
        else:
            sample_values = {
                'alcohol': 13.0,
                'malic_acid': 2.0,
                'ash': 2.0,
                'total_phenols': 2.5,
                'flavanoids': 2.5,
                'color_intensity': 5.0
            }
        
        # Create input fields
        input_data = {}
        
        # Arrange inputs in 2 columns
        input_col1, input_col2 = st.columns(2)
        
        features = list(FEATURE_INFO.keys())
        for idx, feature in enumerate(features):
            info = FEATURE_INFO[feature]
            col = input_col1 if idx % 2 == 0 else input_col2
            
            with col:
                value = st.number_input(
                    f"{info['name']} ({info['unit']})",
                    min_value=float(info['range'][0]),
                    max_value=float(info['range'][1]),
                    value=float(sample_values[feature]),
                    step=0.1,
                    help=info['description'],
                    key=feature
                )
                input_data[feature] = value
        
        # Predict button
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Predict Wine Cultivar", type="primary")
        
        if predict_button:
            # Prepare input data
            features_df = pd.DataFrame([input_data])
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            # Display prediction
            cultivar_info = CULTIVAR_INFO[prediction]
            confidence = prediction_proba[prediction] * 100
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>{cultivar_info['emoji']} Predicted Cultivar</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{cultivar_info['name']}</h1>
                <p style="font-size: 1.2rem;">{cultivar_info['description']}</p>
                <p style="font-size: 1.5rem; margin-top: 1rem;">Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution
            st.subheader("📊 Prediction Probability Distribution")
            prob_df = pd.DataFrame({
                'Cultivar': [f"{CULTIVAR_INFO[i]['emoji']} {CULTIVAR_INFO[i]['name']}" for i in range(3)],
                'Probability (%)': prediction_proba * 100
            })
            st.bar_chart(prob_df.set_index('Cultivar'))
            
            # Show input summary
            with st.expander("📋 View Input Summary"):
                summary_df = pd.DataFrame({
                    'Feature': [FEATURE_INFO[k]['name'] for k in input_data.keys()],
                    'Value': [f"{v:.2f} {FEATURE_INFO[k]['unit']}" for k, v in input_data.items()]
                })
                st.dataframe(summary_df, use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        <div class="feature-card">
        <strong>Step 1:</strong> Enter the chemical property values of your wine sample.
        </div>
        
        <div class="feature-card">
        <strong>Step 2:</strong> Adjust the values using the sliders or input fields.
        </div>
        
        <div class="feature-card">
        <strong>Step 3:</strong> Click the "Predict Wine Cultivar" button.
        </div>
        
        <div class="feature-card">
        <strong>Step 4:</strong> View the predicted cultivar and confidence level.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Tip:** Use the 'Load Sample Wine' button in the sidebar to test with sample data!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Wine Cultivar Prediction System</strong> | Developed by Victor Emeka (23cg034065)</p>
        <p>Machine Learning Algorithm: Random Forest Classifier | Model Persistence: Joblib</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
