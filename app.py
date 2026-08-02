import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- 1. КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Прогноз Солнечной Энергии",
    page_page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. КАСТОМНЫЙ CSS СТИЛЬ ---
st.markdown("""
    <style>
    /* Главный фон и отступы */
    .main {
        background-color: #f8fafc;
    }
    
    /* Карточки метрик */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Заголовки */
    .header-title {
        color: #1e293b;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .header-subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_model():
    # Поиск возможных имен файлов модели
    possible_names = ['model.joblib', 'solar_model.pkl', 'model.pkl', 'solar_energy_model.joblib']
    for name in possible_names:
        if os.path.exists(name):
            try:
                return joblib.load(name), name
            except Exception as e:
                st.warning(f"Ошибка загрузки {name}: {e}")
    return None, None

model, model_filename = load_model()

# --- 4. БОКОВАЯ ПАНЕЛЬ (ВВОД ДАННЫХ) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=70)
st.sidebar.title("☀️ Параметры системы")

# Быстрые пресеты
preset = st.sidebar.selectbox(
    "Загрузить пресет погоды:",
    ["Пользовательские настройки", "Ясный летний день", "Пасмурный день", "Жаркий полдень"]
)

# Дефолтные значения по умолчанию
default_temp = 22.0
default_rad = 750.0
default_hum = 45.0
default_cloud = 10.0
default_cap = 10.0

if preset == "Ясный летний день":
    default_temp, default_rad, default_hum, default_cloud = 25.0, 900.0, 35.0, 5.0
elif preset == "Пасмурный день":
    default_temp, default_rad, default_hum, default_cloud = 15.0, 250.0, 80.0, 85.0
elif preset == "Жаркий полдень":
    default_temp, default_rad, default_hum, default_cloud = 36.0, 1050.0, 25.0, 0.0

st.sidebar.subheader("🌡️ Метеоусловия")
solar_rad = st.sidebar.slider("Солнечная радиация (Вт/м²)", 0.0, 1200.0, float(default_rad), 10.0)
temp = st.sidebar.slider("Температура воздуха (°C)", -20.0, 50.0, float(default_temp), 0.5)
humidity = st.sidebar.slider("Влажность воздуха (%)", 0.0, 100.0, float(default_hum), 1.0)
cloud_cover = st.sidebar.slider("Облачность (%)", 0.0, 100.0, float(default_cloud), 1.0)

st.sidebar.subheader("⚙️ Характеристики станции")
capacity_kw = st.sidebar.number_input("Установленная мощность станций (кВт)", 1.0, 1000.0, float(default_cap), 1.0)

# --- 5. РАСЧЕТ ПРОГНОЗА ---
# Подготовка входного датафрейма для модели
input_df = pd.DataFrame({
    'solar_radiation': [solar_rad],
    'temperature': [temp],
    'humidity': [humidity],
    'cloud_cover': [cloud_cover]
})

# Если модель загружена — используем её, иначе применяем физ-модель
if model is not None:
    try:
        # Проверяем количество фичей в модели
        predicted_power = model.predict(input_df)[0]
    except Exception:
        # Если имена колонок не совпали с теми, на которых обучалась модель
        predicted_power = model.predict(input_df.values)[0]
else:
    # Резервный физический алгоритм расчёта (если файл модели не найден)
    temp_efficiency = 1.0 - max(0.0, (temp - 25.0) * 0.004) # Потеря 0.4% на градус выше 25C
    cloud_loss = 1.0 - (cloud_cover / 100.0 * 0.75)
    predicted_power = capacity_kw * (solar_rad / 1000.0) * temp_efficiency * cloud_loss

# Ограничиваем прогнозируемое значение физическими рамками
predicted_power = max(0.0, min(predicted_power, capacity_kw * 1.05))
efficiency_pct = (predicted_power / capacity_kw) * 100 if capacity_kw > 0 else 0

# --- 6. ГЛАВНАЯ СТРАНИЦА ---
st.markdown("<h1 class='header-title'>Интеллектуальный Прогноз Солнечной Генерации</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-subtitle'>Система прогнозирования выработки электроэнергии на основе машинного обучения</p>", unsafe_allow_html=True)

if model_filename:
    st.success(f"Загружена модель машинного обучения: `{model_filename}`")
else:
    st.info("Используется базовый расчетный модуль. Загрузите файл `model.joblib` в корень репозитория для активации ML-модели.")

st.markdown("---")

# Метрики в верхней панели
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Прогноз Мощности</div>
            <div class='metric-value' style='color:#0284c7;'>{predicted_power:.2f} кВт</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Эффективность</div>
            <div class='metric-value' style='color:#16a34a;'>{efficiency_pct:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    daily_est = predicted_power * 5.2 # Средний эквивалент часов
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Оценка за день</div>
            <div class='metric-value' style='color:#d97706;'>{daily_est:.1f} кВт·ч</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    co2_saved = daily_est * 0.5 # ~0.5 кг CO2 на кВт·ч
    st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Экономия CO₂ / день</div>
            <div class='metric-value' style='color:#059669;'>{co2_saved:.1f} кг</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. ИНТЕРАКТИВНЫЕ ГРАФИКИ ---
tab1, tab2, tab3 = st.tabs(["📊 Индикатор Мощности", "📈 Суточный Профиль", "🔍 Анализ Факторов"])

with tab1:
    # Спидометр (Gauge Chart)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = predicted_power,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Текущая выработка (кВт)", 'font': {'size': 20}},
        delta = {'reference': capacity_kw * 0.7, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [0, capacity_kw], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0284c7"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, capacity_kw * 0.3], 'color': '#fee2e2'},
                {'range': [capacity_kw * 0.3, capacity_kw * 0.7], 'color': '#fef3c7'},
                {'range': [capacity_kw * 0.7, capacity_kw], 'color': '#dcfce7'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': capacity_kw
            }
        }
    ))
    fig_gauge.update_layout(height=380, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    # Симуляция суточного профиля выработки
    hours = np.arange(0, 24)
    # Синусоидальная модель инсоляции по часам суток
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
        title="Ориентировочный суточный график генерации электроэнергии",
        color_discrete_sequence=['#38bdf8']
    )
    fig_line.update_layout(
        xaxis_title="Время суток",
        yaxis_title="Мощность (кВт)",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    # Зависимость генерации от температуры при текущей радиации
    temp_range = np.linspace(-10, 45, 50)
    power_vs_temp = []
    
    for t in temp_range:
        if model is not None:
            try:
                p = model.predict(pd.DataFrame({'solar_radiation': [solar_rad], 'temperature': [t], 'humidity': [humidity], 'cloud_cover': [cloud_cover]}))[0]
            except:
                p = predicted_power
        else:
            t_eff = 1.0 - max(0.0, (t - 25.0) * 0.004)
            p = capacity_kw * (solar_rad / 1000.0) * t_eff * (1.0 - cloud_cover / 100.0 * 0.75)
        power_vs_temp.append(max(0.0, p))
        
    df_sens = pd.DataFrame({'Температура (°C)': temp_range, 'Прогнозируемая Мощность (кВт)': power_vs_temp})
    
    fig_sens = px.line(
        df_sens, 
        x='Температура (°C)', 
        y='Прогнозируемая Мощность (кВт)',
        title=f"Влияние температуры на выработку (при инсоляции {solar_rad} Вт/м²)",
        color_discrete_sequence=['#f59e0b']
    )
    fig_sens.add_vline(x=temp, line_dash="dash", line_color="red", annotation_text="Текущее значение")
    fig_sens.update_layout(template="plotly_white")
    st.plotly_chart(fig_sens, use_container_width=True)

# --- 8. ЭКСПОРТ ДАННЫХ ---
st.markdown("---")
col_exp1, col_exp2 = st.columns([3, 1])

with col_exp1:
    st.caption("Данные пересчитываются автоматически при изменении любого из параметров в боковой панели.")

with col_exp2:
    export_data = input_df.copy()
    export_data['predicted_power_kw'] = predicted_power
    export_data['efficiency_pct'] = efficiency_pct
    csv_data = export_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Скачать результат (CSV)",
        data=csv_data,
        file_name="solar_prediction_result.csv",
        mime="text/csv"
    )