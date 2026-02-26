import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Recomendador de Cursos — CAIXA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta Institucional
CA_AZUL = "#0070B8"
CA_ESCURO = "#003F8A"
CA_LARANJA = "#F5A623"
CA_VERDE = "#00A859"
CA_CINZA = "#333333"
CA_CLARO = "#F0F7FF"
CA_BRANCO = "#FFFFFF"

# CSS
st.markdown("""
<style>

/* Reset */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #F0F7FF !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Tipografia */
h1, h2, h3 { color: #003F8A !important; }

/* ⚠️ REMOVIDO span daqui */
p, label, div { color: #333333 !important; }

/* Banner branco */
.banner-white, .banner-white * {
    color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e0e0e0;
}

/* Botão PRIMARY - versão compatível Streamlit atual */
button[kind="primary"] {
    background: linear-gradient(90deg, #F5A623 0%, #E08B00 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.6rem 1rem;
    box-shadow: 0 4px 12px rgba(245,166,35,0.3);
}

button[kind="primary"] span {
    color: #FFFFFF !important;
}

button[kind="primary"]:hover {
    box-shadow: 0 6px 16px rgba(245,166,35,0.4);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #0070B8 !important;
}

.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    border-radius: 8px 8px 0 0;
    border: 1px solid #0070B8 !important;
    border-bottom: none !important;
}

.stTabs [aria-selected="true"] p {
    color: #0070B8 !important;
}

</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_models():
    return joblib.load("modelos_comparacao.pkl")

modelos_dict = load_models()

# Sidebar
st.sidebar.markdown("### ⚙️ Painel de Controle")
modelo_selecionado = st.sidebar.radio(
    "Algoritmo Ativo:",
    ["Gradient Boosting", "Random Forest", "Regressão Logística"]
)

if modelo_selecionado == "Gradient Boosting":
    model_ativo = modelos_dict["GB"]
    nome_tecnico = "Gradient Boosting"
    cor_tema = CA_AZUL
elif modelo_selecionado == "Random Forest":
    model_ativo = modelos_dict["RF"]
    nome_tecnico = "Random Forest"
    cor_tema = CA_VERDE
else:
    model_ativo = modelos_dict["LR"]
    nome_tecnico = "Regressão Logística"
    cor_tema = CA_CINZA

classes = model_ativo.named_steps["model"].classes_

# Header
st.markdown(f"""
<div class="banner-white"
     style="background:linear-gradient(135deg,{cor_tema} 0%,{CA_ESCURO} 100%);
            padding:32px;border-radius:12px;margin-bottom:24px;">
  <h1 style="margin:0;">🏦 Recomendador de Trilhas de IA</h1>
  <p style="margin-top:8px;">Modelo Ativo: {nome_tecnico}</p>
</div>
""", unsafe_allow_html=True)

# Tabs
aba1, aba2 = st.tabs(["📋 Perfil", "🎯 Recomendação"])

with aba1:
    area = st.selectbox("Área", ["TI","Riscos","Operações"])
    funcao = st.selectbox("Função", ["Analista","Gestor"])
    tempo_de_casa = st.slider("Tempo de casa", 0, 40, 5)

    if st.button("GERAR RECOMENDAÇÃO PERSONALIZADA 🚀", type="primary", use_container_width=True):
        st.success("Perfil processado com sucesso!")

with aba2:
    st.info("Recomendação aparecerá aqui.")
