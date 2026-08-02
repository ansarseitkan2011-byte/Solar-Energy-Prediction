import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- 1. КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="AI Прогноз Солнечной Станции",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. КАСТОМНЫЙ CSS СТИЛЬ ---
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value { font-size: 30px; font-weight: 700; color: #0f172a; }
    .metric-label { font-size: 13px; color: #64748b; text-transform: uppercase; font-weight: 600; }
    .header-title { color: #1e293b; font-weight: 800; margin-bottom: 0px; }
    .header-subtitle { color: #64748b; font-size: 15px; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    possible_names = ['model.pkl', 'model.joblib', 'solar_model.pkl']
    for name in possible_names:
        if os.path.exists(name):
            try:
                m = joblib.load(name)
                return m, name
            except Exception:
                pass
    return None, None

model, model_filename = load_model()

# --- 4. БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.title("☀️ Параметры погоды")

preset = st.sidebar.selectbox(
    "Загрузить пресет погоды:",
    ["Пользовательские настройки", "Ясный летний день", "Пасмурный день", "Жаркий полдень"]
)

default_rad = 800.0
default_mod_temp = 35.0
default_air_temp = 25.0

if preset == "Ясный летний день":
    default_rad, default_mod_temp, default_air_temp = 950.0, 42.0, 30.0
elif preset == "Пасмурный день":
    default_rad, default_mod_temp, default_air_temp = 200.0, 18.0, 15.0
elif preset == "Жаркий полдень":
    default_rad, default_mod_temp, default_air_temp = 1100.0, 52.0, 38.0

solar_rad = st.sidebar.slider("Солнечная радиация (Вт/м²)", 0.0, 1200.0, float(default_rad), 10.0)
module_temp = st.sidebar.slider("Температура модуля (°C)", 0.0, 80.0, float(default_mod_temp), 0.5)
ambient_temp = st.sidebar.slider("Температура воздуха (°C)", -10.0, 50.0, float(default_air_temp), 0.5)

# --- 5. ПРЯМОЙ ФИЗИКО-МАТЕМАТИЧЕСКИЙ И AI РАСЧЕТ ---
irradiance_norm = solar_rad / 1000.0  # Инсоляция (0...1.2)
temp_loss = 1.0 - max(0.0, (module_temp - 25.0) * 0.004) # Потеря от температуры

# Базовый расчет физики панели (для станции 750 кВт)
base_power = irradiance_norm * 750.0 * temp_loss

ai_factor = 1.0

if model is not None:
    try:
        # Пробуем получить коэффициент коррекции от загруженной AI-модели
        inp = np.array([[irradiance_norm, module_temp, ambient_temp]])
        pred = float(model.predict(inp)[0])
        if pred > 0:
            ai_factor = min(1.3, max(0.7, pred))
    except Exception:
        pass

# Итоговая мощность с учетом AI и параметров
predicted_power = max(0.0, base_power * ai_factor)

# --- 6. ГЛАВНАЯ СТРАНИЦА ---
st.markdown("<h1 class='header-title'>☀️ Прогнозирование выработки солнечной энергии</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Меняй параметры погоды на панели слева, чтобы ИИ рассчитал выработку энергии в реальном времени.</p>", unsafe_allow_html=True)

if model_filename:
    st.success(f"✅ AI-модель `{model_filename}` успешно загружена!")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Результат прогноза AI</div>
            <div class='metric-value' style='color:#0284c7;'>{predicted_power:,.2f} кВт</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    daily_est = predicted_power * 5.2
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Оценка за день</div>
            <div class='metric-value' style='color:#d97706;'>{daily_est:,.1f} кВт·ч</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    co2_saved = daily_est * 0.5
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Экономия CO₂ / день</div>
            <div class='metric-value' style='color:#059669;'>{co2_saved:,.1f} кг</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. ГРАФИКИ ---
tab1, tab2, tab3 = st.tabs(["📊 Индикатор Мощности", "📈 Суточный Профиль", "🔍 Анализ Инсоляции"])

with tab1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_power,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Выработка энергии (DC Power)", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 800]},
            'bar': {'color': "#0284c7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 250], 'color': '#fee2e2'},
                {'range': [250, 550], 'color': '#fef3c7'},
                {'range': [550, 800], 'color': '#dcfce7'}
            ]
        }
    ))
    fig_gauge.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    hours = np.arange(0, 24)
    sun_profile = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
    hourly_power = predicted_power * sun_profile
    
    df_hourly = pd.DataFrame({
        'Час суток': [f"{h:02d}:00" for h in hours],
        'Выработка (кВт)': hourly_power
    })
    
    fig_line = px.area(
        df_hourly, 
        x='Час суток', 
        y='Выработка (кВт)',
        title="Суточный профиль генерации",
        color_discrete_sequence=['#38bdf8']
    )
    fig_line.update_layout(template="plotly_white")
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    rad_range = np.linspace(0.0, 1200.0, 25)
    power_vs_rad = [(r / 1000.0) * 750.0 * temp_loss * ai_factor for r in rad_range]
    
    df_rad = pd.DataFrame({'Солнечная радиация (Вт/м²)': rad_range, 'Выработка (кВт)': power_vs_rad})
    
    fig_rad = px.line(
        df_rad, 
        x='Солнечная радиация (Вт/м²)', 
        y='Выработка (кВт)',
        title="Зависимость выработки от уровня инсоляции",
        color_discrete_sequence=['#f59e0b']
    )
    fig_rad.add_vline(x=solar_rad, line_dash="dash", line_color="red", annotation_text="Текущая точка")
    fig_rad.update_layout(template="plotly_white")
    st.plotly_chart(fig_rad, use_container_width=True)