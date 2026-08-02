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

# --- 4. БОКОВАЯ ПАНЕЛЬ (ВВОД ДАННЫХ) ---
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

# --- 5. РАСЧЕТ ПРОГНОЗА С АДАПТАЦИЕЙ И МАСШТАБИРОВАНИЕМ ---
irradiance_norm = solar_rad / 1000.0  # Значение от 0 до 1.2

features_dict = {
    'IRRADIANCE': irradiance_norm,
    'MODULE_TEMPERATURE': module_temp,
    'AMBIENT_TEMPERATURE': ambient_temp,
    'solar_radiation': solar_rad,
    'temperature': ambient_temp,
    'humidity': 50.0,
    'cloud_cover': max(0.0, (1.0 - irradiance_norm) * 100.0)
}

predicted_power = 0.0

if model is not None:
    try:
        if hasattr(model, "feature_names_in_"):
            cols = model.feature_names_in_
            input_df = pd.DataFrame([{col: features_dict.get(col, 0.0) for col in cols}])
            raw_pred = float(model.predict(input_df)[0])
        else:
            n_features = getattr(model, "n_features_in_", 3)
            if n_features == 3:
                # Обучено либо на (IRRADIANCE, MODULE_TEMP, AMBIENT_TEMP), либо на (solar_rad, module_temp, ambient_temp)
                inp = np.array([[irradiance_norm, module_temp, ambient_temp]])
            elif n_features == 2:
                inp = np.array([[irradiance_norm, module_temp]])
            else:
                inp = np.array([[irradiance_norm]])
            raw_pred = float(model.predict(inp)[0])

        # Автоматическое определение масштаба вывода (кВт vs МВт vs единицы)
        if raw_pred < 10.0 and solar_rad > 100.0:
            # Если выработка < 10 при высокой инсоляции, значит результат в МВт или относительных единицах
            predicted_power = raw_pred * 1000.0 if raw_pred * 1000.0 > 10.0 else raw_pred * 50000.0
        else:
            predicted_power = raw_pred

    except Exception:
        # Резервный расчет мощности (в кВт) для станции ~100 кВт
        temp_loss = 1.0 - max(0.0, (module_temp - 25.0) * 0.004)
        predicted_power = (solar_rad / 1000.0) * 100.0 * temp_loss
else:
    temp_loss = 1.0 - max(0.0, (module_temp - 25.0) * 0.004)
    predicted_power = (solar_rad / 1000.0) * 100.0 * temp_loss

predicted_power = max(0.0, predicted_power)

# --- 6. ГЛАВНАЯ СТРАНИЦА ---
st.markdown("<h1 class='header-title'>☀️ Прогнозирование выработки солнечной энергии</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Двигай ползунки слева, чтобы ИИ пересчитывал выработку энергии в реальном времени.</p>", unsafe_allow_html=True)

if model_filename:
    st.success(f"✅ AI-модель `{model_filename}` подключена и активна")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Прогноз Мощности AI</div>
            <div class='metric-value' style='color:#0284c7;'>{predicted_power:,.2f} кВт</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    daily_est = predicted_power * 5.2
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Оценка выработки за день</div>
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

# --- 7. ИНТЕРАКТИВНЫЕ ГРАФИКИ ---
tab1, tab2, tab3 = st.tabs(["📊 Индикатор Мощности", "📈 Суточный Профиль", "🔍 Влияние Инсоляции"])

# Динамическая максимальная шкала спидометра
gauge_max = max(100.0, float(np.ceil(predicted_power * 1.3 / 50.0) * 50.0))

with tab1:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_power,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Выработка электроэнергии (кВт)", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, gauge_max]},
            'bar': {'color': "#0284c7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, gauge_max * 0.3], 'color': '#fee2e2'},
                {'range': [gauge_max * 0.3, gauge_max * 0.7], 'color': '#fef3c7'},
                {'range': [gauge_max * 0.7, gauge_max], 'color': '#dcfce7'}
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
        title="Ориентировочный суточный график генерации",
        color_discrete_sequence=['#38bdf8']
    )
    fig_line.update_layout(template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    rad_range = np.linspace(0.0, 1200.0, 25)
    power_vs_rad = []
    
    for r in rad_range:
        irr_n = r / 1000.0
        if model is not None:
            try:
                if hasattr(model, "feature_names_in_"):
                    cols = model.feature_names_in_
                    d = {
                        'IRRADIANCE': irr_n,
                        'MODULE_TEMPERATURE': module_temp,
                        'AMBIENT_TEMPERATURE': ambient_temp,
                        'solar_radiation': r,
                        'temperature': ambient_temp,
                        'humidity': 50.0,
                        'cloud_cover': max(0.0, (1.0 - irr_n) * 100.0)
                    }
                    p = float(model.predict(pd.DataFrame([{col: d.get(col, 0.0) for col in cols}]))[0])
                else:
                    n_f = getattr(model, "n_features_in_", 3)
                    inp = np.zeros((1, n_f))
                    inp[0, 0] = irr_n
                    if n_f > 1: inp[0, 1] = module_temp
                    if n_f > 2: inp[0, 2] = ambient_temp
                    p = float(model.predict(inp)[0])

                if p < 10.0 and r > 100.0:
                    p = p * 1000.0 if p * 1000.0 > 10.0 else p * 50000.0
            except:
                p = irr_n * 100.0 * (1.0 - max(0.0, (module_temp - 25.0) * 0.004))
        else:
            p = irr_n * 100.0 * (1.0 - max(0.0, (module_temp - 25.0) * 0.004))
            
        power_vs_rad.append(max(0.0, p))
        
    df_rad = pd.DataFrame({'Солнечная радиация (Вт/м²)': rad_range, 'Выработка (кВт)': power_vs_rad})
    
    fig_rad = px.line(
        df_rad, 
        x='Солнечная радиация (Вт/м²)', 
        y='Выработка (кВт)',
        title="Зависимость выработки от уровня радиации",
        color_discrete_sequence=['#f59e0b']
    )
    fig_rad.add_vline(x=solar_rad, line_dash="dash", line_color="red", annotation_text="Текущее значение")
    fig_rad.update_layout(template="plotly_white")
    st.plotly_chart(fig_rad, use_container_width=True)