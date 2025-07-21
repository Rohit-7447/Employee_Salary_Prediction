import streamlit as st
import joblib
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def validate_model(model):
    """
    Validate that the loaded object is actually a model
    """
    if model is None:
        return False
    
    # Check if it has predict method
    if not hasattr(model, 'predict'):
        st.error(f"❌ Loaded object is {type(model).__name__}, not a model!")
        return False
    
    # Try a simple prediction test
    try:
        # Test with dummy data
        test_input = np.array([[5, 1, 0, 0]])  # experience=5, education=1, dept=0, location=0
        test_pred = model.predict(test_input)
        if len(test_pred) > 0:
            return True
        else:
            return False
    except Exception as e:
        st.error(f"❌ Model validation failed: {str(e)}")
        return False

def safe_load_model(model_path):
    """
    Safely load model with multiple fallback methods and validation
    """
    try:
        # Method 1: Try joblib first
        model = joblib.load(model_path)
        
        # Validate the loaded model
        if validate_model(model):
            st.success("✅ Model loaded successfully with joblib!")
            return model, "joblib"
        else:
            st.warning("⚠️ Joblib loaded an invalid object, trying pickle...")
            
    except AttributeError as e:
        st.warning(f"⚠️ Joblib failed due to version mismatch: {str(e)[:100]}...")
    except Exception as e:
        st.warning(f"⚠️ Joblib failed: {str(e)[:100]}...")
        
    try:
        # Method 2: Try pickle as fallback
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Validate the loaded model
        if validate_model(model):
            st.success("✅ Model loaded successfully with pickle!")
            return model, "pickle"
        else:
            st.error("❌ Pickle loaded an invalid object!")
            return None, "failed"
            
    except Exception as e:
        st.error(f"❌ Pickle also failed: {str(e)[:100]}...")
        return None, "failed"

def create_dummy_model():
    """
    Create a simple dummy model for demonstration if loading fails
    """
    st.warning("🔧 Creating dummy model for demonstration...")
    
    # Create sample data for training a simple model
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic training data
    experience = np.random.randint(0, 30, n_samples)
    education_level = np.random.choice([0, 1, 2], n_samples)  # 0=Bachelor, 1=Master, 2=PhD
    department = np.random.choice([0, 1, 2, 3], n_samples)   # 0=IT, 1=HR, 2=Finance, 3=Marketing
    location = np.random.choice([0, 1, 2], n_samples)        # 0=City1, 1=City2, 2=City3
    
    # Generate realistic salary based on features
    salary = (30000 + 
              experience * 2000 + 
              education_level * 5000 + 
              department * 3000 + 
              location * 4000 + 
              np.random.normal(0, 5000, n_samples))
    
    # Ensure positive salaries
    salary = np.maximum(salary, 25000)
    
    # Create and train model
    X = np.column_stack([experience, education_level, department, location])
    
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X, salary)
    
    st.info("🎯 Dummy model trained on synthetic data for demonstration")
    return model, "dummy"

@st.cache_resource
def load_prediction_model():
    """
    Load the model with caching for better performance
    Always returns (model, method) tuple consistently
    """
    model_path = "best_model.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        st.info("📁 Please ensure the model file is in the same directory as this app")
        return create_dummy_model()
    
    model, method = safe_load_model(model_path)
    
    if model is None or not validate_model(model):
        st.error("❌ Could not load a valid trained model!")
        st.info("🔄 Using dummy model instead...")
        return create_dummy_model()
    
    return model, method

def main():
    # Page configuration
    st.set_page_config(
        page_title="Employee Salary Prediction",
        page_icon="💼",
        layout="wide"
    )
    
    # Title and description
    st.title("💼 Employee Salary Prediction System")
    st.markdown("---")
    st.markdown("### Predict employee salaries using Machine Learning")
    
    # Load model - this now always returns a tuple
    model, load_method = load_prediction_model()
    
    # Display model status with proper validation
    col1, col2, col3 = st.columns(3)
    with col1:
        # Now we can safely check if model exists and is valid
        if model is not None and hasattr(model, 'predict'):
            st.metric("Model Status", "✅ Loaded")
        else:
            st.metric("Model Status", "❌ Failed")
            
    with col2:
        st.metric("Loading Method", load_method.title())
        
    with col3:
        if model is not None and hasattr(model, 'n_estimators'):
            st.metric("Model Type", f"Gradient Boosting ({model.n_estimators} trees)")
        else:
            st.metric("Model Type", "Unknown")
    
    st.markdown("---")
    
    # Only show prediction interface if model is valid
    if model is None or not hasattr(model, 'predict'):
        st.error("❌ Cannot make predictions without a valid model!")
        st.info("Please resolve the model loading issue or check your model file.")
        return
    
    # Input section
    st.subheader("📝 Enter Employee Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        experience = st.slider(
            "Years of Experience", 
            min_value=0, 
            max_value=65, 
            value=5,
            help="Total years of professional experience"
        )
        
        education = st.selectbox(
            "Education Level",
            options=["Bachelor's Degree", "Master's Degree", "PhD"],
            help="Highest educational qualification"
        )
    
    with col2:
        department = st.selectbox(
            "Department",
            options=["Information Technology", "Human Resources", "Finance", "Marketing"],
            help="Employee's working department"
        )
        
        location = st.selectbox(
            "Work Location",
            options=["Tier 1 City", "Tier 2 City", "Tier 3 City"],
            help="Office location category"
        )
    
    # Encode categorical variables
    education_map = {"Bachelor's Degree": 0, "Master's Degree": 1, "PhD": 2}
    department_map = {"Information Technology": 0, "Human Resources": 1, "Finance": 2, "Marketing": 3}
    location_map = {"Tier 1 City": 0, "Tier 2 City": 1, "Tier 3 City": 2}
    
    education_encoded = education_map[education]
    department_encoded = department_map[department]
    location_encoded = location_map[location]
    
    # Prediction section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Predict Salary", type="primary", use_container_width=True):
            try:
                # Prepare input data
                input_data = np.array([[experience, education_encoded, department_encoded, location_encoded]])
                
                # Make prediction - now we're sure model has predict method
                prediction = model.predict(input_data)
                predicted_salary = prediction[0]
                
                # Display results
                st.success("✨ Prediction Complete!")
                
                # Results display
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.metric(
                        label="💰 Predicted Annual Salary",
                        value=f"₹{predicted_salary:,.0f}",
                        delta=f"₹{predicted_salary/12:,.0f}/month"
                    )
                
                # Additional insights
                st.markdown("---")
                st.subheader("📊 Prediction Insights")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Experience Impact:** {experience} years contributes significantly to salary")
                    st.info(f"**Education Bonus:** {education} adds premium to base salary")
                
                with col2:
                    st.info(f"**Department Factor:** {department} role influences compensation")
                    st.info(f"**Location Adjustment:** {location} affects salary standards")
                
                # Confidence note
                if load_method == "dummy":
                    st.warning("⚠️ **Note:** This prediction is from a demo model. For production use, please resolve the model loading issue.")
                else:
                    st.success("✅ **Confidence:** High - Using production-trained model")
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Please check your input values and try again.")
    
    # Footer information
    st.markdown("---")
    st.markdown("### 🔧 Troubleshooting")
    
    with st.expander("Having model loading issues?"):
        st.markdown("""
        **Common solutions:**
        1. **Version Mismatch:** The model was trained with a different scikit-learn version
        2. **Solution 1:** Install matching scikit-learn version: `pip install scikit-learn==1.3.0`
        3. **Solution 2:** Retrain the model with your current environment
        4. **Solution 3:** Use the dummy model for testing (currently active if main model fails)
        
        **Current Status:** Model loaded using **{0}** method
        """.format(load_method))
    
    with st.expander("About this Application"):
        st.markdown("""
        This application uses **Gradient Boosting Algorithm** to predict employee salaries based on:
        - Years of Experience
        - Education Level  
        - Department
        - Work Location
        
        **Model Features:**
        - Handles non-linear relationships
        - Considers feature interactions
        - Provides reliable predictions
        - Built using scikit-learn
        """)

if __name__ == "__main__":
    main()