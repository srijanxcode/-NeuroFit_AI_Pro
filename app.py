import pandas as pd
import numpy as np
import datetime as dt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from transformers import pipeline
from streamlit_echarts import st_echarts
import warnings
warnings.filterwarnings('ignore')

# ======================
# INITIALIZATION
# ======================
st.set_page_config(
    page_title="NeuroFit AI Health",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Enhanced Theme
def set_theme():
    st.markdown(f"""
    <style>
        .main {{
            background-color: #0E1117;
            color: #FAFAFA;
            font-family: 'Arial', sans-serif;
        }}
        .sidebar .sidebar-content {{
            background-color: #1A1C23;
        }}
        .stTextInput input, .stNumberInput input, .stSelectbox select {{
            background-color: #1A1C23 !important;
            color: white !important;
            border-radius: 8px;
        }}
        .st-bb {{ background-color: transparent; }}
        .st-at {{ background-color: #4B4E58; }}
        div[data-testid="stMetric"] {{
            background-color: #1A1C23;
            border-radius: 10px;
            padding: 5% 5% 5% 10%;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .stProgress > div > div > div > div {{
            background-image: linear-gradient(to right, #00D4FF, #0072FF);
        }}
        section[data-testid="stSidebar"] {{
            width: 350px !important;
        }}
        h1, h2, h3 {{
            color: #00D4FF;
        }}
        .stButton>button {{
            border-radius: 8px;
            background: linear-gradient(to right, #00D4FF, #0072FF);
            color: white;
            font-weight: bold;
        }}
    </style>
    """, unsafe_allow_html=True)

set_theme()

# Enhanced QA Model with fallback responses
@st.cache_resource
def load_qa_model():
    try:
        return pipeline(
            "question-answering", 
            model="distilbert-base-cased-distilled-squad",
            framework="pt"
        )
    except:
        return None

health_qa = load_qa_model()

# ======================
# ENHANCED COMPONENTS
# ======================
class BioNeuralNetwork(nn.Module):
    """Advanced neural network for health predictions"""
    def __init__(self, input_size):
        super().__init__()
        self.bio_lstm = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.bio_lstm(x)
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return self.predictor(context_vector)

class EnhancedHealthSimulator:
    """Generates more comprehensive synthetic health data"""
    def __init__(self):
        self.base_date = dt.datetime.now() - dt.timedelta(days=30)
        self.health_conditions = [
            "Normal", "Elevated HR", "Low Oxygen", 
            "High Stress", "Poor Sleep", "Inactive"
        ]
        
    def generate_vitals(self, age, weight, activity_level):
        dates = [self.base_date + dt.timedelta(days=i) for i in range(30)]
        
        # Activity level modifiers
        activity_modifier = {
            "Sedentary": 0.8,
            "Lightly Active": 1.0,
            "Moderately Active": 1.2,
            "Very Active": 1.5
        }.get(activity_level, 1.0)
        
        # Age modifiers
        age_modifier = 1 + (age - 30) * 0.01
        
        return pd.DataFrame({
            'Date': dates,
            'HeartRate': np.clip(np.cumsum(np.random.normal(0, 2, 30)) + 72 * age_modifier, 60, 100),
            'BloodOxygen': np.clip(np.random.normal(98, 0.5, 30), 95, 100),
            'StressLevel': np.clip(np.abs(np.cumsum(np.random.normal(0, 3, 30))), 0, 100),
            'SleepQuality': np.clip(np.random.normal(80, 10, 30), 50, 100),
            'ActivityScore': np.clip(np.cumsum(np.random.normal(0, 5, 30)) * activity_modifier, 0, 100),
            'CaloriesBurned': np.clip(2000 + np.random.normal(0, 200, 30) * activity_modifier, 1500, 3500),
            'Steps': np.clip(np.random.normal(8000, 2000, 30) * activity_modifier, 0, 20000),
            'WaterIntake': np.clip(np.random.normal(2000, 300, 30), 1000, 4000),
            'Condition': random.choices(self.health_conditions, k=30)
        })

# Enhanced Data Engine
class EnhancedHealthDataEngine:
    def __init__(self, age, weight, activity_level):
        self.simulator = EnhancedHealthSimulator()
        self.vitals_data = self.simulator.generate_vitals(age, weight, activity_level)
        
    def get_health_forecast(self, days=7):
        future_dates = [self.vitals_data['Date'].iloc[-1] + dt.timedelta(days=i) for i in range(1, days+1)]
        return pd.DataFrame({
            'Date': future_dates,
            'HeartRate': np.clip(self.vitals_data['HeartRate'].iloc[-1] + np.random.normal(0, 1, days), 60, 100),
            'BloodOxygen': np.clip(self.vitals_data['BloodOxygen'].iloc[-1] + np.random.normal(0, 0.2, days), 95, 100),
            'StressLevel': np.clip(self.vitals_data['StressLevel'].iloc[-1] + np.random.normal(0, 2, days), 0, 100),
            'SleepQuality': np.clip(self.vitals_data['SleepQuality'].iloc[-1] + np.random.normal(0, 3, days), 50, 100),
            'ActivityScore': np.clip(self.vitals_data['ActivityScore'].iloc[-1] + np.random.normal(2, 3, days), 0, 100),
            'CaloriesBurned': np.clip(self.vitals_data['CaloriesBurned'].iloc[-1] + np.random.normal(50, 30, days), 1500, 3500),
            'Steps': np.clip(self.vitals_data['Steps'].iloc[-1] + np.random.normal(200, 100, days), 0, 20000),
            'WaterIntake': np.clip(self.vitals_data['WaterIntake'].iloc[-1] + np.random.normal(50, 20, days), 1000, 4000)
        })

# Enhanced Visualization Engine
class EnhancedNeuroViz:
    @staticmethod
    def render_health_dashboard(data):
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "📈 Trends", "🥗 Nutrition", "🎮 Fun"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(EnhancedNeuroViz.render_metric_pie(data), use_container_width=True)
            with col2:
                st.plotly_chart(EnhancedNeuroViz.render_health_clock(data), use_container_width=True)
            
            st.plotly_chart(EnhancedNeuroViz.render_correlation_matrix(data), use_container_width=True)
        
        with tab2:
            st.plotly_chart(EnhancedNeuroViz.render_health_timeline(data), use_container_width=True)
            st.plotly_chart(EnhancedNeuroViz.render_3d_health_map(data), use_container_width=True)
        
        with tab3:
            st.plotly_chart(EnhancedNeuroViz.render_nutrition_chart(data), use_container_width=True)
        
        with tab4:
            st.plotly_chart(EnhancedNeuroViz.render_fitness_game(data), use_container_width=True)

    @staticmethod
    def render_metric_pie(data):
        conditions = data['Condition'].value_counts()
        fig = px.pie(conditions, values=conditions.values, names=conditions.index,
                    title="Health Condition Distribution", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False)
        return fig

    @staticmethod
    def render_health_clock(data):
        fig = make_subplots(rows=1, cols=4, specs=[[{'type': 'domain'}] * 4])
        
        metrics = [
            ('HeartRate', '❤️', 'rgb(255, 99, 132)'),
            ('BloodOxygen', '🫁', 'rgb(54, 162, 235)'),
            ('StressLevel', '🧠', 'rgb(255, 206, 86)'),
            ('ActivityScore', '🏃', 'rgb(75, 192, 192)')
        ]
        
        for i, (metric, emoji, color) in enumerate(metrics, 1):
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=data[metric].iloc[-1],
                domain={'row': 0, 'column': i-1},
                title={'text': f"{emoji} {metric}"},
                gauge={
                    'axis': {'range': [None, 100] if metric in ['StressLevel', 'ActivityScore'] else [
                        data[metric].min()-5, 
                        data[metric].max()+5
                    ]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgray'},
                        {'range': [30, 70], 'color': 'gray'},
                        {'range': [70, 100], 'color': 'darkgray'}
                    ],
                }
            ), row=1, col=i)
            
        fig.update_layout(height=250, margin=dict(t=0, b=0))
        return fig

    @staticmethod
    def render_health_timeline(data):
        fig = go.Figure()
        
        metrics = ['HeartRate', 'BloodOxygen', 'StressLevel', 'SleepQuality', 'ActivityScore']
        colors = px.colors.qualitative.Plotly
        
        for i, col in enumerate(metrics):
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data[col],
                name=col,
                line=dict(width=2, color=colors[i]),
                mode='lines+markers'
            ))
            
        fig.update_layout(
            title="Biometric Timeline",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified",
            template="plotly_dark",
            height=400
        )
        return fig

    @staticmethod
    def render_3d_health_map(data):
        fig = px.scatter_3d(
            data,
            x='HeartRate',
            y='StressLevel',
            z='SleepQuality',
            color='ActivityScore',
            size='BloodOxygen',
            hover_name='Date',
            title="3D Health Constellation",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig.update_layout(
            scene=dict(
                xaxis_title='Heart Rate',
                yaxis_title='Stress Level',
                zaxis_title='Sleep Quality'
            ),
            height=600
        )
        return fig

    @staticmethod
    def render_correlation_matrix(data):
        numeric_cols = data.select_dtypes(include=np.number).columns
        corr = data[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                       color_continuous_scale=px.colors.diverging.RdBu_r,
                       title="Health Metrics Correlation")
        return fig

    @staticmethod
    def render_nutrition_chart(data):
        avg_calories = data['CaloriesBurned'].mean()
        water_intake = data['WaterIntake'].mean()
        
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
        
        fig.add_trace(go.Pie(
            labels=['Protein', 'Carbs', 'Fats'],
            values=[0.3*avg_calories/4, 0.4*avg_calories/4, 0.3*avg_calories/9],
            name="Macros",
            hole=0.4
        ), 1, 1)
        
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=water_intake,
            title={'text': "Water Intake (ml)"},
            gauge={
                'axis': {'range': [0, 4000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2000], 'color': "lightgray"},
                    {'range': [2000, 3000], 'color': "gray"},
                    {'range': [3000, 4000], 'color': "darkgray"}
                ],
            }
        ), 1, 2)
        
        fig.update_layout(title_text="Nutrition Dashboard", showlegend=True)
        return fig

    @staticmethod
    def render_fitness_game(data):
        steps_goal = 10000
        steps_progress = min(data['Steps'].iloc[-1] / steps_goal * 100, 100)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=steps_progress,
            title={'text': "Step Challenge Progress"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': steps_progress
                }
            }
        ))
        fig.update_layout(height=300)
        return fig

# Enhanced AI Assistant with fallback responses
class EnhancedNeuroAssistant:
    def __init__(self, qa_model, health_data, age, weight, activity_level):
        self.qa = qa_model
        self.data = health_data
        self.age = age
        self.weight = weight
        self.activity_level = activity_level
        self.context = self._create_context()
        self.fallback_responses = [
            "Based on your health data, I recommend focusing on improving your sleep quality first.",
            "Your activity levels look good, but you might want to monitor your stress levels.",
            "I notice your heart rate is slightly elevated. Consider some relaxation techniques.",
            "Your hydration seems adequate, but could be improved with an extra glass of water daily.",
            "Your biometrics suggest you're doing well overall, with room for improvement in activity."
        ]
        
    def _create_context(self):
        last_week = self.data.iloc[-7:]
        return f"""
        User Profile:
        - Age: {self.age}
        - Weight: {self.weight} kg
        - Activity Level: {self.activity_level}
        
        Last 7 Days Averages:
        - Heart Rate: {last_week['HeartRate'].mean():.1f} bpm
        - Blood Oxygen: {last_week['BloodOxygen'].mean():.1f}%
        - Stress Level: {last_week['StressLevel'].mean():.1f}/100
        - Sleep Quality: {last_week['SleepQuality'].mean():.1f}/100
        - Activity Score: {last_week['ActivityScore'].mean():.1f}/100
        - Steps: {last_week['Steps'].mean():.0f} per day
        - Water Intake: {last_week['WaterIntake'].mean():.0f} ml
        
        Current Health Condition: {self.data['Condition'].iloc[-1]}
        """
        
    def generate_diet_plan(self):
        calories = self.data['CaloriesBurned'].mean()
        protein = (0.3 * calories) / 4  # 30% of calories from protein
        carbs = (0.4 * calories) / 4    # 40% from carbs
        fats = (0.3 * calories) / 9     # 30% from fats
        
        meals = {
            "Breakfast": "Oatmeal with nuts and berries",
            "Lunch": "Grilled chicken with quinoa and vegetables",
            "Dinner": "Salmon with sweet potato and greens",
            "Snacks": "Greek yogurt or handful of almonds"
        }
        
        if self.activity_level == "Very Active":
            meals["Pre-Workout"] = "Banana with peanut butter"
            meals["Post-Workout"] = "Protein shake with banana"
        
        return {
            "Daily Calories": f"{calories:.0f} kcal",
            "Macros": {
                "Protein": f"{protein:.1f}g",
                "Carbs": f"{carbs:.1f}g",
                "Fats": f"{fats:.1f}g"
            },
            "Meal Plan": meals,
            "Hydration Goal": f"{self.data['WaterIntake'].mean():.0f} ml water"
        }
    
    def query(self, question):
        if not self.qa:
            return random.choice(self.fallback_responses)
        
        try:
            result = self.qa(
                question=question,
                context=self.context,
                max_answer_len=200
            )
            return result['answer']
        except:
            # Enhanced fallback with context-aware responses
            if "diet" in question.lower() or "nutrition" in question.lower():
                diet_plan = self.generate_diet_plan()
                response = "Here's a personalized diet plan based on your data:\n\n"
                response += f"Calories: {diet_plan['Daily Calories']}\n"
                response += f"Protein: {diet_plan['Macros']['Protein']}\n"
                response += f"Carbs: {diet_plan['Macros']['Carbs']}\n"
                response += f"Fats: {diet_plan['Macros']['Fats']}\n\n"
                response += "Meal Suggestions:\n"
                for meal, desc in diet_plan['Meal Plan'].items():
                    response += f"- {meal}: {desc}\n"
                return response
            elif "exercise" in question.lower() or "workout" in question.lower():
                return f"Based on your {self.activity_level} activity level, I recommend 30-45 minutes of moderate exercise daily."
            elif "sleep" in question.lower():
                return "Your sleep quality could be improved by maintaining a consistent sleep schedule and reducing screen time before bed."
            else:
                return random.choice(self.fallback_responses)

# ======================
# ENHANCED MAIN APP
# ======================
def main():
    st.sidebar.title("🧠 NeuroFit AI Pro")
    st.sidebar.markdown("Configure your health profile")
    
    # Enhanced User Inputs
    with st.sidebar:
        age = st.slider("Age", 18, 80, 30)
        weight = st.slider("Weight (kg)", 40, 150, 70)
        height = st.slider("Height (cm)", 140, 220, 170)
        activity_level = st.selectbox(
            "Activity Level",
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"]
        )
        health_goal = st.selectbox(
            "Primary Health Goal",
            ["Weight Loss", "Muscle Gain", "Stress Reduction", "Better Sleep", "General Fitness"]
        )
        
        st.markdown("---")
        st.markdown("### Prediction Settings")
        forecast_days = st.slider("Forecast Period (days)", 1, 14, 7)
        st.markdown("---")
        
        if st.button("🔮 Generate Health Analysis", key="analyze"):
            with st.spinner("Analyzing your health data..."):
                time.sleep(1)
                st.session_state.analyzed = True
        
        if st.button("🔄 Refresh Data", key="refresh"):
            st.session_state.clear()
    
    # Initialize engines with enhanced parameters
    if 'engine' not in st.session_state:
        st.session_state.engine = EnhancedHealthDataEngine(age, weight, activity_level)
    
    if 'viz' not in st.session_state:
        st.session_state.viz = EnhancedNeuroViz()
    
    if 'assistant' not in st.session_state:
        st.session_state.assistant = EnhancedNeuroAssistant(
            health_qa, 
            st.session_state.engine.vitals_data,
            age,
            weight,
            activity_level
        )
    
    # Main Dashboard
    st.title("🧠 NeuroFit AI Pro - Health Dashboard")
    st.markdown("### Your Personalized Health Insights")
    
    # Health Dashboard
    st.session_state.viz.render_health_dashboard(st.session_state.engine.vitals_data)
    
    # Forecast Section
    if st.session_state.get('analyzed', False):
        st.subheader(f"🔮 {forecast_days}-Day Health Forecast")
        forecast = st.session_state.engine.get_health_forecast(forecast_days)
        st.plotly_chart(st.session_state.viz.render_health_timeline(
            pd.concat([st.session_state.engine.vitals_data, forecast])
        ), use_container_width=True)
    
    # Enhanced AI Assistant
    st.markdown("---")
    st.subheader("🤖 NeuroFit AI Assistant Pro")
    
    # Display diet plan in an expander
    with st.expander("🍽️ View Personalized Diet Plan"):
        diet_plan = st.session_state.assistant.generate_diet_plan()
        st.write("**Daily Nutrition Targets:**")
        st.write(f"- Calories: {diet_plan['Daily Calories']}")
        st.write(f"- Protein: {diet_plan['Macros']['Protein']}")
        st.write(f"- Carbs: {diet_plan['Macros']['Carbs']}")
        st.write(f"- Fats: {diet_plan['Macros']['Fats']}")
        
        st.write("\n**Recommended Meals:**")
        for meal, desc in diet_plan['Meal Plan'].items():
            st.write(f"- **{meal}**: {desc}")
        
        st.write(f"\n**Hydration Goal**: {diet_plan['Hydration Goal']}")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your NeuroFit AI assistant. How can I help you with your health today?"}
        ]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your health..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = st.session_state.assistant.query(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Health Challenges
    st.markdown("---")
    st.subheader("🏆 Daily Health Challenges")
    
    challenges = {
        "💧 Hydration": "Drink 3L of water today",
        "🚶 Steps": "Reach 10,000 steps",
        "🧘 Stress": "Practice 10 mins of meditation",
        "💤 Sleep": "Get 7-8 hours of quality sleep"
    }
    
    for challenge, desc in challenges.items():
        with st.expander(f"{challenge}: {desc}"):
            progress = st.slider(f"Progress for {challenge}", 0, 100, random.randint(30, 80))
            st.metric("Completion", f"{progress}%")

if __name__ == "__main__":
    main()

