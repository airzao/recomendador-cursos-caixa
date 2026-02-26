import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

st.set_page_config(
    page_title="Recomendador de Cursos — CAIXA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

CA_AZUL    = "#0070B8"
CA_ESCURO  = "#003F8A"
CA_LARANJA = "#F5A623"
CA_VERDE   = "#00A859"
CA_CINZA   = "#333333"
CA_CLARO   = "#F0F7FF"
CA_BRANCO  = "#FFFFFF"

# ═══════════════════════════════════════════
# CSS GLOBAL
# ═══════════════════════════════════════════
st.markdown("""
<style>

/* FUNDO */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
section.main, .main {
    background-color: #F0F7FF !important;
    color: #333333 !important;
}

/* ===== BANNERS COM TEXTO BRANCO FORÇADO ===== */
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

/* Labels */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #333333 !important;
    font-weight: 600 !important;
}

/* Botão */
[data-testid="baseButton-primary"] {
    background-color: #0070B8 !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════
st.markdown(f"""
<div class="banner-white"
     style="background:linear-gradient(135deg,{CA_AZUL} 0%,{CA_ESCURO} 100%);
            padding:36px 40px;border-radius:12px;margin-bottom:28px;
            box-shadow:0 6px 24px rgba(0,63,138,0.25);">
  <h1 style="margin:0;font-size:2.4rem;font-weight:900;">
    🏦 Recomendador de Trilhas de IA
  </h1>
  <p style="margin:8px 0 0 0;font-size:1.05rem;opacity:0.9;">
    Caixa Econômica Federal | Sistema Inteligente de Recomendação |
    <span style="background:{CA_LARANJA};
                 padding:2px 10px;border-radius:12px;font-size:0.85rem;font-weight:700;">
        ML • Stacking Classifier
    </span>
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("modelo_recomendacao_curso.pkl")

model = load_model()
classes = model.named_steps["model"].classes_

aba1, aba2, aba3, aba4 = st.tabs(
    ["📋 Perfil", "🎯 Recomendação Premium", "📊 Modelo & Métricas", "🎓 Para o Professor"]
)

# ══════════════════════════════════════════════════════════════════════
# ABA 1 — PERFIL
# ══════════════════════════════════════════════════════════════════════
with aba1:

    col1, col2 = st.columns(2)

    with col1:
        area = st.selectbox("Área", ["Agencia","TI","Riscos","Financeiro"])
        funcao = st.selectbox("Função", ["Analista","Gestor"])
        tempo = st.slider("Tempo de casa", 0, 40, 5)

    with col2:
        ja_ia = st.radio("Já usou IA?", ["Sim","Nao"], horizontal=True)
        nivel = st.selectbox("Nível programação", ["Nenhum","Basico","Intermediario","Avancado"])

    if st.button("GERAR RECOMENDAÇÃO", type="primary", use_container_width=True):
        st.session_state["inp"] = pd.DataFrame([{
            "area":area,
            "funcao":funcao,
            "tempo_de_casa":tempo,
            "ja_utilizou_ia":ja_ia,
            "nivel_programacao":nivel
        }])
        st.success("Perfil salvo!")

# ══════════════════════════════════════════════════════════════════════
# ABA 2 — TOP 3
# ══════════════════════════════════════════════════════════════════════
with aba2:

    if "inp" not in st.session_state:
        st.info("Preencha o perfil primeiro.")
    else:

        proba = model.predict_proba(st.session_state["inp"])[0]
        top_idx = np.argsort(-proba)[:3]
        medalhas = ["🥇","🥈","🥉"]

        fundos = [
            f"linear-gradient(135deg,{CA_AZUL} 0%,{CA_ESCURO} 100%)",
            f"linear-gradient(135deg,{CA_LARANJA} 0%,#E08B00 100%)",
            f"linear-gradient(135deg,#4A7C59 0%,#2D5A3D 100%)"
        ]

        for i, idx in enumerate(top_idx):
            curso = classes[idx]
            conf  = proba[idx] * 100

            st.markdown(f"""
<div class="banner-white"
     style="background:{fundos[i]};
            padding:28px 32px;border-radius:14px;
            margin-bottom:14px;
            box-shadow:0 8px 24px rgba(0,63,138,0.18);">
  <h3 style="margin:0 0 10px 0;font-size:1.35rem;font-weight:800;">
      {medalhas[i]} {curso}
  </h3>
  <p style="font-size:1rem;font-weight:700;">{conf:.1f}%</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ABA 3 — KPIs
# ══════════════════════════════════════════════════════════════════════
with aba3:

    m1, m2, m3, m4 = st.columns(4)

    kpis = [
        (m1, "92,8%", "Accuracy", CA_AZUL, CA_ESCURO),
        (m2, "92,4%", "Macro F1", CA_LARANJA,"#E08B00"),
        (m3, "99,0%", "Top-3", CA_VERDE,"#007A40"),
        (m4, "9.493", "Empregados", "#7B2D8B","#4A1A55"),
    ]

    for col_, val_, lbl_, c1_, c2_ in kpis:
        with col_:
            st.markdown(f"""
<div class="banner-white"
     style="background:linear-gradient(135deg,{c1_} 0%,{c2_} 100%);
            padding:22px;border-radius:12px;text-align:center;
            box-shadow:0 4px 16px rgba(0,0,0,0.12);">
  <h1 style="margin:0;font-size:2.2rem;font-weight:900;">{val_}</h1>
  <p style="margin:8px 0 0 0;font-size:0.9rem;font-weight:600;">{lbl_}</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ABA 4 — RESUMO EXECUTIVO
# ══════════════════════════════════════════════════════════════════════
with aba4:

    st.markdown(f"""
<div class="banner-white"
     style="background:linear-gradient(135deg,{CA_AZUL} 0%,{CA_ESCURO} 100%);
            padding:28px 32px;border-radius:14px;margin-bottom:24px;">
  <h3 style="margin:0 0 12px 0;font-size:1.4rem;">Resumo Executivo</h3>
  <p style="margin:0;font-size:1rem;line-height:1.7;">
    Sistema inteligente de recomendação de trilhas de IA
    para empregados da Caixa Econômica Federal.
    Acurácia de 92,8% e Top-3 de 99%.
  </p>
</div>
""", unsafe_allow_html=True)
