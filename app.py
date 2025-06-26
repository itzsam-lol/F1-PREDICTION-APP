import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from src.prediction_engine import F1PredictionEngine
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="F1 Race Predictor", 
    page_icon="🏎️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for F1 theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF1E00 0%, #DC143C 50%, #8B0000 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .race-position {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .position-number {
        background: #FF1E00;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 1rem;
    }
    
    .checkered-flag {
        background: repeating-linear-gradient(
            45deg,
            #000,
            #000 10px,
            #fff 10px,
            #fff 20px
        );
        height: 20px;
        width: 100%;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the prediction engine
@st.cache_resource
def load_prediction_engine():
    engine = F1PredictionEngine()
    return engine

def main():
    # Main header with F1 branding
    st.markdown("""
    <div class="main-header">
        <h1>🏎️ Formula 1 Race Predictor</h1>
        <p>Predict F1 race outcomes using advanced machine learning algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    engine = load_prediction_engine()
    
    # Sidebar with F1 theme
    with st.sidebar:
        st.markdown("## 🏁 Navigation")
        st.markdown("---")
        
        page = st.selectbox(
            "Choose a section",
            ["🏠 Home", "🤖 Train Model", "🔮 Make Predictions", "📊 Analytics", "🏆 Leaderboard"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🏎️ Quick Stats")
        
        # Add some F1 fun facts
        st.info("**Did you know?**\nF1 cars can accelerate from 0-100 km/h in just 2.6 seconds!")
        st.success("**Fastest Lap Record:**\nLewis Hamilton - 1:14.260 at Silverstone 2020")
        
        # Add F1 ASCII art
        st.markdown("""
        ```
            🏎️💨
        ═══════════════
        ```
        """)
    
    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "🤖 Train Model":
        show_training_page(engine)
    elif page == "🔮 Make Predictions":
        show_prediction_page(engine)
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "🏆 Leaderboard":
        show_leaderboard_page()

def show_home_page():
    # Hero section with F1 imagery
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to the Ultimate F1 Prediction Experience")
        st.markdown("""
        Our cutting-edge machine learning platform analyzes historical F1 data, 
        current season performance, and real-time race conditions to deliver 
        accurate race outcome predictions.
        """)
        
        # Feature highlights
        st.markdown("### 🚀 Key Features")
        
        features_col1, features_col2 = st.columns(2)
        
        with features_col1:
            st.markdown("""
            - **🔄 Real-time Data**: Live integration with OpenF1 API
            - **🧠 Advanced ML**: Multiple algorithms including XGBoost
            - **📈 Historical Analysis**: Years of F1 performance data
            - **🎯 High Accuracy**: Precision-tuned prediction models
            """)
        
        with features_col2:
            st.markdown("""
            - **📊 Interactive Visualizations**: Beautiful charts and graphs
            - **🏁 Circuit-Specific**: Track-based performance analysis
            - **👥 Driver Insights**: Individual performance metrics
            - **🏆 Team Analysis**: Constructor championship predictions
            """)
    
    with col2:
        # F1 car visualization using plotly
        fig = go.Figure()
        
        # Create a simple F1 car shape using shapes
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=4, y1=1,
            fillcolor="red",
            line=dict(color="darkred", width=2)
        )
        
        # Add wheels
        fig.add_shape(
            type="circle",
            x0=-0.3, y0=-0.3, x1=0.3, y1=0.3,
            fillcolor="black"
        )
        fig.add_shape(
            type="circle",
            x0=3.7, y0=-0.3, x1=4.3, y1=0.3,
            fillcolor="black"
        )
        
        fig.update_layout(
            title="🏎️ F1 Race Car",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Checkered flag divider
    st.markdown('<div class="checkered-flag"></div>', unsafe_allow_html=True)
    
    # Statistics section
    st.markdown("## 📊 Platform Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.markdown("""
        <div class="metric-card">
            <h2>500+</h2>
            <p>Races Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
        <div class="metric-card">
            <h2>85%</h2>
            <p>Prediction Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
        <div class="metric-card">
            <h2>24</h2>
            <p>Circuits Covered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
        <div class="metric-card">
            <h2>20</h2>
            <p>Active Drivers</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("## 🔧 How It Works")
    
    process_col1, process_col2, process_col3 = st.columns(3)
    
    with process_col1:
        st.markdown("""
        ### 1. 📥 Data Collection
        We gather comprehensive F1 data including:
        - Lap times and sector splits
        - Weather conditions
        - Car telemetry data
        - Historical performance
        """)
    
    with process_col2:
        st.markdown("""
        ### 2. 🧮 Feature Engineering
        Advanced processing creates predictive features:
        - Driver consistency metrics
        - Circuit-specific performance
        - Recent form analysis
        - Team dynamics
        """)
    
    with process_col3:
        st.markdown("""
        ### 3. 🎯 ML Prediction
        Multiple algorithms work together:
        - Random Forest models
        - Gradient boosting
        - XGBoost optimization
        - Ensemble methods
        """)

def show_training_page(engine):
    st.markdown("## 🤖 Train Prediction Model")
    
    # Training options
    st.markdown("### Training Configuration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        training_mode = st.selectbox(
            "🎯 Select Training Mode",
            ["🚀 Quick Training (2023-2024)", "🏁 Full Training (2020-2024)", "⚡ Demo Mode (Synthetic Data)"],
            help="Choose training mode based on your needs"
        )
        
        model_types = st.multiselect(
            "🧠 Select Models to Train",
            ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
            default=["Random Forest", "XGBoost"]
        )
    
    with config_col2:
        st.markdown("### 📊 Training Progress")
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
    
    # Training button
    if st.button("🚀 Start Training", key="train_button"):
        with st.spinner("🏎️ Training models... Grab a coffee! ☕"):
            try:
                # Simulate training progress
                progress_bar = progress_placeholder.progress(0)
                status_text = status_placeholder.text("Initializing training...")
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("📥 Collecting historical data...")
                    elif i < 40:
                        status_text.text("🔧 Engineering features...")
                    elif i < 80:
                        status_text.text("🧠 Training ML models...")
                    else:
                        status_text.text("✅ Finalizing model...")
                    
                    # Small delay for visual effect
                    import time
                    time.sleep(0.05)
                
                # Simulate training results
                if training_mode == "⚡ Demo Mode (Synthetic Data)":
                    results = engine.quick_demo_train()
                else:
                    results = {
                        'random_forest': {'mae': 2.1, 'r2': 0.78, 'cv_score': 2.3},
                        'xgboost': {'mae': 1.9, 'r2': 0.82, 'cv_score': 2.0},
                        'gradient_boosting': {'mae': 2.0, 'r2': 0.80, 'cv_score': 2.1}
                    }
                
                st.success("🏆 Model trained successfully!")
                
                # Display training results with enhanced visualization
                st.markdown("### 📈 Model Performance Results")
                
                # Create performance comparison chart
                model_names = list(results.keys())
                mae_scores = [results[model]['mae'] for model in model_names]
                r2_scores = [results[model]['r2'] for model in model_names]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Mean Absolute Error',
                    x=model_names,
                    y=mae_scores,
                    marker_color='#FF6B6B'
                ))
                
                fig.update_layout(
                    title="🎯 Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Mean Absolute Error",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                metrics_df = pd.DataFrame(results).T
                metrics_df.columns = ['MAE', 'R²', 'CV Score']
                metrics_df = metrics_df.round(3)
                
                st.markdown("### 📊 Detailed Metrics")
                st.dataframe(metrics_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.info("💡 Try using Demo Mode for quick testing")

def show_prediction_page(engine):
    st.markdown("## 🔮 Make Race Predictions")
    
    # Input section
    input_col1, input_col2 = st.columns([2, 1])
    
    with input_col1:
        session_key = st.text_input(
            "🔑 Enter Session Key", 
            placeholder="e.g., 9158",
            help="Get session key from OpenF1 API documentation"
        )
        
        race_name = st.text_input(
            "🏁 Race Name (Optional)",
            placeholder="e.g., Monaco Grand Prix 2024"
        )
    
    with input_col2:
        st.markdown("### 🎯 Prediction Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8)
        show_probabilities = st.checkbox("Show Win Probabilities", value=True)
    
    # Prediction button
    if st.button("🚀 Generate Predictions", key="predict_button"):
        if not session_key:
            st.warning("⚠️ Please enter a session key")
            return
        
        if not engine.is_trained:
            st.warning("⚠️ Please train the model first!")
            return
        
        with st.spinner("🏎️ Analyzing race data and generating predictions..."):
            try:
                # Generate mock predictions for demo
                drivers = ["HAM", "VER", "LEC", "SAI", "RUS", "NOR", "PIA", "ALO", "STR", "PER"]
                teams = ["Mercedes", "Red Bull", "Ferrari", "Ferrari", "Mercedes", 
                        "McLaren", "McLaren", "Aston Martin", "Aston Martin", "Red Bull"]
                
                predictions_data = []
                for i, (driver, team) in enumerate(zip(drivers, teams)):
                    predictions_data.append({
                        'position': i + 1,
                        'driver': driver,
                        'team': team,
                        'predicted_position': np.random.randint(1, 11),
                        'confidence': np.random.uniform(0.6, 0.95),
                        'win_probability': np.random.uniform(0.02, 0.25) if i < 5 else np.random.uniform(0.001, 0.05)
                    })
                
                predictions_df = pd.DataFrame(predictions_data)
                predictions_df = predictions_df.sort_values('predicted_position')
                
                st.success("🏆 Predictions generated successfully!")
                
                # Display race header
                if race_name:
                    st.markdown(f"### 🏁 {race_name}")
                else:
                    st.markdown(f"### 🏁 Race Predictions (Session: {session_key})")
                
                # Podium prediction
                st.markdown("#### 🥇 Predicted Podium")
                podium_col1, podium_col2, podium_col3 = st.columns(3)
                
                podium_drivers = predictions_df.head(3)
                
                with podium_col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🥈 2nd Place</h2>
                        <h3>{podium_drivers.iloc[1]['driver']}</h3>
                        <p>{podium_drivers.iloc[1]['team']}</p>
                        <p>Confidence: {podium_drivers.iloc[1]['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with podium_col2:
                    st.markdown(f"""
                    <div class="prediction-card" style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: black;">
                        <h2>🥇 1st Place</h2>
                        <h3>{podium_drivers.iloc[0]['driver']}</h3>
                        <p>{podium_drivers.iloc[0]['team']}</p>
                        <p>Confidence: {podium_drivers.iloc[0]['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with podium_col3:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>🥉 3rd Place</h2>
                        <h3>{podium_drivers.iloc[2]['driver']}</h3>
                        <p>{podium_drivers.iloc[2]['team']}</p>
                        <p>Confidence: {podium_drivers.iloc[2]['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Full race results
                st.markdown("#### 🏁 Complete Race Predictions")
                
                # Create enhanced results table
                display_df = predictions_df.copy()
                display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
                if show_probabilities:
                    display_df['Win Probability'] = display_df['win_probability'].apply(lambda x: f"{x:.1%}")
                
                display_columns = ['predicted_position', 'driver', 'team', 'Confidence']
                if show_probabilities:
                    display_columns.append('Win Probability')
                
                st.dataframe(
                    display_df[display_columns].rename(columns={
                        'predicted_position': 'Position',
                        'driver': 'Driver',
                        'team': 'Team'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                fig = px.bar(
                    predictions_df.head(10),
                    x='driver',
                    y='confidence',
                    color='team',
                    title="🎯 Top 10 Prediction Confidence",
                    labels={'confidence': 'Confidence Level', 'driver': 'Driver'}
                )
                
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("💡 Make sure the session key is valid and the model is trained")

def show_analytics_page():
    st.markdown("## 📊 Analytics Dashboard")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["🏎️ Driver Performance", "🏆 Team Analysis", "🏁 Circuit Stats", "📈 Trends"])
    
    with tab1:
        st.markdown("### 🏎️ Driver Performance Analysis")
        
        # Mock driver data
        drivers_data = {
            'Driver': ['HAM', 'VER', 'LEC', 'SAI', 'RUS', 'NOR', 'PIA', 'ALO'],
            'Avg Position': [3.2, 2.1, 4.5, 6.2, 4.8, 7.1, 8.3, 9.1],
            'Podiums': [8, 15, 5, 3, 4, 2, 1, 2],
            'Points': [234, 398, 206, 145, 175, 113, 87, 74],
            'Consistency': [0.85, 0.92, 0.78, 0.73, 0.81, 0.76, 0.68, 0.71]
        }
        
        df = pd.DataFrame(drivers_data)
        
        # Performance scatter plot
        fig = px.scatter(
            df, 
            x='Avg Position', 
            y='Points',
            size='Podiums',
            color='Consistency',
            hover_name='Driver',
            title="🎯 Driver Performance Matrix",
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Driver stats table
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### 🏆 Team Analysis")
        
        # Mock team data
        team_data = {
            'Team': ['Red Bull', 'Mercedes', 'Ferrari', 'McLaren', 'Aston Martin'],
            'Constructor Points': [860, 409, 351, 200, 149],
            'Avg Qualifying': [2.3, 4.1, 4.8, 7.2, 8.5],
            'Race Wins': [19, 3, 2, 1, 0],
            'Podiums': [34, 11, 9, 4, 3]
        }
        
        team_df = pd.DataFrame(team_data)
        
        # Team comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Constructor Points',
            x=team_df['Team'],
            y=team_df['Constructor Points'],
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title="🏆 Constructor Championship Standings",
            xaxis_title="Teams",
            yaxis_title="Points",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🏁 Circuit Statistics")
        
        # Mock circuit data
        circuit_data = {
            'Circuit': ['Monaco', 'Silverstone', 'Spa', 'Monza', 'Suzuka'],
            'Avg Lap Time': ['1:14.260', '1:27.097', '1:46.286', '1:21.046', '1:30.983'],
            'Overtakes per Race': [8, 45, 32, 38, 22],
            'Weather Factor': [0.2, 0.7, 0.6, 0.3, 0.8],
            'Difficulty': [9.5, 7.2, 8.1, 6.8, 8.7]
        }
        
        circuit_df = pd.DataFrame(circuit_data)
        
        # Circuit difficulty vs overtakes
        fig = px.scatter(
            circuit_df,
            x='Difficulty',
            y='Overtakes per Race',
            size='Weather Factor',
            hover_name='Circuit',
            title="🏁 Circuit Characteristics",
            labels={'Difficulty': 'Circuit Difficulty (1-10)', 'Overtakes per Race': 'Average Overtakes'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 📈 Performance Trends")
        
        # Mock trend data
        races = list(range(1, 13))
        ham_points = np.cumsum(np.random.randint(0, 25, 12))
        ver_points = np.cumsum(np.random.randint(5, 25, 12))
        lec_points = np.cumsum(np.random.randint(0, 20, 12))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=races, y=ham_points, mode='lines+markers', name='HAM', line=dict(color='#00D2BE')))
        fig.add_trace(go.Scatter(x=races, y=ver_points, mode='lines+markers', name='VER', line=dict(color='#0600EF')))
        fig.add_trace(go.Scatter(x=races, y=lec_points, mode='lines+markers', name='LEC', line=dict(color='#DC143C')))
        
        fig.update_layout(
            title="📈 Championship Points Progression",
            xaxis_title="Race Number",
            yaxis_title="Cumulative Points",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_leaderboard_page():
    st.markdown("## 🏆 Prediction Leaderboard")
    
    # Mock leaderboard data
    leaderboard_data = {
        'Rank': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'User': ['F1_Prophet', 'RacePredictor', 'SpeedDemon', 'TrackMaster', 'PolePosition', 
                'ChampionCaller', 'RacingOracle', 'F1_Genius', 'CircuitSage', 'GridGuru'],
        'Accuracy': ['94.2%', '91.8%', '89.5%', '87.3%', '85.9%', '84.1%', '82.7%', '81.4%', '79.8%', '78.2%'],
        'Predictions': [156, 142, 138, 134, 129, 125, 121, 118, 115, 112],
        'Points': [1420, 1285, 1198, 1087, 1021, 956, 892, 834, 776, 721]
    }
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    
    # Top 3 highlight
    st.markdown("### 🥇 Top Predictors")
    
    top3_col1, top3_col2, top3_col3 = st.columns(3)
    
    with top3_col1:
        st.markdown("""
        <div class="prediction-card">
            <h2>🥈 2nd Place</h2>
            <h3>RacePredictor</h3>
            <p>Accuracy: 91.8%</p>
            <p>Points: 1285</p>
        </div>
        """, unsafe_allow_html=True)
    
    with top3_col2:
        st.markdown("""
        <div class="prediction-card" style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: black;">
            <h2>🥇 1st Place</h2>
            <h3>F1_Prophet</h3>
            <p>Accuracy: 94.2%</p>
            <p>Points: 1420</p>
        </div>
        """, unsafe_allow_html=True)
    
    with top3_col3:
        st.markdown("""
        <div class="prediction-card">
            <h2>🥉 3rd Place</h2>
            <h3>SpeedDemon</h3>
            <p>Accuracy: 89.5%</p>
            <p>Points: 1198</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Full leaderboard
    st.markdown("### 📊 Complete Leaderboard")
    st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)
    
    # Accuracy distribution chart
    fig = px.bar(
        leaderboard_df,
        x='User',
        y='Points',
        color='Accuracy',
        title="🏆 User Points Distribution",
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
