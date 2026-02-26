# ================================
# IMPORTS
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ================================
# CONFIGURAÇÃO DA PÁGINA
# ================================
st.set_page_config(
    page_title="Plataforma de Direcionamento Estratégico",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================
# PALETA INSTITUCIONAL
# ================================
CA_AZUL    = "#0070B8"
CA_ESCURO  = "#003F8A"
CA_LARANJA = "#F5A623"
CA_VERDE   = "#00A859"
CA_CINZA   = "#333333"
CA_CLARO   = "#F0F7FF"
CA_BRANCO  = "#FFFFFF"


# ================================
# CSS GLOBAL
# ================================
st.markdown("""
<style>

/* Fundo geral */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"],
section.main, .main {
    background-color: #F0F7FF !important;
    color: #333333 !important;
}

/* Texto branco dentro dos banners */
.banner-white,
.banner-white h1,
.banner-white h2,
.banner-white h3,
.banner-white h4,
.banner-white p,
.banner-white span,
.banner-white b,
.banner-white strong {
    color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #dee2e6;
}

/* Tabs */
.stTabs[data-baseweb="tab-list"] {
    border-bottom: 3px solid #0070B8 !important;
}

/* Tabela HTML customizada */
.custom-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 0.95rem;
    font-family: sans-serif;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
}
.custom-table thead tr {
    background-color: #0070B8;
    color: #ffffff;
}
.custom-table th, .custom-table td {
    padding: 12px 15px;
    border: 1px solid #dddddd;
    color: #333333;
    vertical-align: top;
}
.custom-table tbody tr:nth-of-type(even) {
    background-color: #f3f3f3;
}

</style>
""", unsafe_allow_html=True)


# ================================
# LOAD MODELOS
# ================================
@st.cache_resource
def load_models():
    return joblib.load("modelos_comparacao.pkl")

modelos_dict = load_models()


# ================================
# SIDEBAR
# ================================
st.sidebar.markdown(
    f"<h2 style='color:{CA_AZUL};font-weight:800;'>🛠️ Simulador de Modelos</h2>",
    unsafe_allow_html=True
)

modelo_selecionado = st.sidebar.radio(
    "Modelo Ativo:",
    ["Gradient Boosting", "Random Forest", "Regressão Logística"]
)

if modelo_selecionado == "Gradient Boosting":
    model_ativo = modelos_dict["GB"]
    cor_tema = CA_AZUL
elif modelo_selecionado == "Random Forest":
    model_ativo = modelos_dict["RF"]
    cor_tema = CA_VERDE
else:
    model_ativo = modelos_dict["LR"]
    cor_tema = CA_CINZA

classes = model_ativo.named_steps["model"].classes_


# ================================
# HEADER
# ================================
st.markdown(f"""
<div class="banner-white"
     style="background:linear-gradient(135deg,{cor_tema} 0%,{CA_ESCURO} 100%);
            padding:36px 40px;border-radius:12px;margin-bottom:28px;">
  <h1 style="margin:0;font-size:2.3rem;font-weight:900;">
    Plataforma de Direcionamento Estratégico de Capacitação
  </h1>
</div>
""", unsafe_allow_html=True)


# ================================
# TABS
# ================================
aba1, aba2, aba3, aba4 = st.tabs(
    ["📋 Perfil", "🎯 Recomendação Premium", "📊 Modelo & Métricas", "🎓 Detalhes do Projeto"]
)


# ================================
# ABA 4 — DETALHES DO PROJETO
# ================================
with aba4:

    st.markdown(f"<h2 style='color:{CA_AZUL};'>📄 Detalhes do Projeto</h2>", unsafe_allow_html=True)

    st.markdown(f"""
<div class="banner-white"
     style="background:linear-gradient(135deg,{cor_tema} 0%,{CA_ESCURO} 100%);
            padding:28px 32px;border-radius:14px;margin-bottom:24px;">
  <h3 style="margin:0 0 12px 0;font-size:1.4rem;">📚 Resumo Executivo</h3>
  <p style="margin:0;font-size:1rem;line-height:1.7;">
    Sistema inteligente de recomendação de trilhas de Inteligência Artificial
    para os empregados da Caixa Econômica Federal.
    A PoC comparou três algoritmos:
    Regressão Logística, Random Forest e Gradient Boosting.
    O vencedor foi o Gradient Boosting com 93,4% de acurácia
    e 97,7% de Top-3 Accuracy.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<table class="custom-table">
  <thead>
    <tr>
      <th style="width: 20%;">Aspecto</th>
      <th>Detalhe</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Problema</b></td>
      <td>Dificuldade de direcionar corretamente trilhas de IA para diferentes perfis funcionais.</td>
    </tr>
    <tr>
      <td><b>Objetivo</b></td>
      <td>Acelerar adoção segura e estratégica de IA.</td>
    </tr>
    <tr>
      <td><b>Solução</b></td>
      <td>Modelo supervisionado baseado em 9 features funcionais e comportamentais.</td>
    </tr>
    <tr>
      <td><b>ROI</b></td>
      <td>Redução de desperdício em treinamento e aumento de efetividade.</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🧬 As 9 Features do Modelo")
    st.markdown("""
| # | Feature | Tipo |
|---|---------|------|
|1|Área|Categórica|
|2|Função|Categórica|
|3|Tempo de Casa|Numérica|
|4|Já utilizou IA|Binária|
|5|Atividade Principal|Categórica|
|6|Objetivo 6 meses|Categórica|
|7|Impacto do erro|Categórica|
|8|Forma de uso|Categórica|
|9|Nível programação|Categórica|
""")
