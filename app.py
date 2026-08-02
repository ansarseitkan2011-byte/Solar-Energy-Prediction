import streamlit as st
import joblib
import pandas as pd

# 1. Настройка страницы приложения
st.set_page_config(
    page_title="AI Прогноз Солнечной Станции", 
    page_icon="☀️", 
    layout="centered"
)

# Заголовок и описание
st.title("☀️ Прогнозирование выработки солнечной энергии")
st.write("Меняй параметры погоды на панели слева, чтобы ИИ рассчитал выработку энергии в реальном времени.")

# 2. Функция загрузки сохраненной модели ИИ
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    model = load_model()
    st.success("✅ AI-модель успешно загружена!")
except Exception as e:
    st.error("❌ Не найден файл model.pkl. Сначала запусти train_ai.py!")
    st.stop()

st.markdown("---")

# 3. Боковая панель с интерактивными ползунками
st.sidebar.header("🎛 Настройки погоды")

irrad = st.sidebar.slider(
    "Солнечная инсоляция (IRRADIATION)", 
    min_value=0.0, 
    max_value=1.5, 
    value=0.5, 
    step=0.05
)

mod_temp = st.sidebar.slider(
    "Температура модуля (°C)", 
    min_value=10.0, 
    max_value=70.0, 
    value=35.0, 
    step=1.0
)

amb_temp = st.sidebar.slider(
    "Температура воздуха (°C)", 
    min_value=10.0, 
    max_value=50.0, 
    value=25.0, 
    step=1.0
)

# 4. Формирование данных для модели
input_data = pd.DataFrame({
    'AMBIENT_TEMPERATURE': [amb_temp],
    'MODULE_TEMPERATURE': [mod_temp],
    'IRRADIATION': [irrad]
})

# 5. Получение предсказания от ИИ
prediction = model.predict(input_data)[0]

# 6. Отображение результатов на экране
st.subheader("📊 Результат прогноза AI:")

st.metric(
    label="Прогнозируемая выработка (DC Power)", 
    value=f"{prediction:,.2f} кВт"
)

# Интерактивный статус
if prediction == 0:
    st.info("🌙 Ночь или нет солнца. Выработка равна нулю.")
elif prediction < 100000:
    st.warning("☁️ Низкая выработка (облачно или раннее утро/вечер).")
else:
    st.success("🔥 Высокая выработка! Солнечная станция работает на мощном режиме.")