import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.prediction_engine import F1PredictionEngine

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

# Initialize the prediction engine
@st.cache_resource
def load_prediction_engine():
    engine = F1PredictionEngine()
    return engine

def main():
    st.title("🏎️ Formula 1 Race Prediction App")
    st.markdown("Predict F1 race outcomes using machine learning and real-time data!")
    
    engine = load_prediction_engine()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Train Model", "Make Predictions", "Analytics"])
    
    if page == "Home":
        show_home_page()
    elif page == "Train Model":
        show_training_page(engine)
    elif page == "Make Predictions":
        show_prediction_page(engine)
    elif page == "Analytics":
        show_analytics_page()

def show_home_page():
    st.header("Welcome to F1 Race Predictor")
    st.write("""
    This application uses machine learning to predict Formula 1 race outcomes based on:
    - Historical performance data
    - Current season statistics
    - Circuit-specific performance
    - Recent form and consistency
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Features")
        st.write("- Real-time data from OpenF1 API")
        st.write("- Multiple ML models (Random Forest, XGBoost, etc.)")
        st.write("- Historical data analysis")
        st.write("- Interactive visualizations")
    
    with col2:
        st.subheader("How to Use")
        st.write("1. Train the model with historical data")
        st.write("2. Enter session key for upcoming race")
        st.write("3. Get predictions with confidence intervals")
        st.write("4. Analyze results and trends")

def show_training_page(engine):
    st.header("Train Prediction Model")
    
    if st.button("Start Training"):
        with st.spinner("Training model... This may take a few minutes."):
            try:
                results = engine.train_prediction_model()
                st.success("Model trained successfully!")
                
                # Display training results
                st.subheader("Model Performance")
                for model_name, metrics in results.items():
                    st.write(f"**{model_name}**")
                    st.write(f"- MAE: {metrics['mae']:.3f}")
                    st.write(f"- R²: {metrics['r2']:.3f}")
                    st.write(f"- CV Score: {metrics['cv_score']:.3f}")
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def show_prediction_page(engine):
    st.header("Make Race Predictions")
    
    session_key = st.text_input("Enter Session Key", help="Get session key from OpenF1 API")
    
    if st.button("Predict Race Outcome") and session_key:
        if not engine.is_trained:
            st.warning("Please train the model first!")
            return
        
        with st.spinner("Making predictions..."):
            try:
                predictions = engine.predict_upcoming_race(session_key)
                
                st.subheader("Predicted Race Results")
                st.dataframe(predictions)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=predictions.head(10), x='predicted_position', y='driver', ax=ax)
                ax.set_title("Top 10 Predicted Finishing Positions")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def show_analytics_page():
    st.header("Analytics Dashboard")
    st.write("Analytics features would be implemented here")
    # Add charts, statistics, and analysis tools

if __name__ == "__main__":
    main()
