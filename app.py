import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="Recomendador de Cursos — CAIXA", page_icon="🏦", layout="wide", initial_sidebar_state="collapsed")

CAIXA_GREEN, CAIXA_DARK, CAIXA_LIGHT, CAIXA_GRAY, CAIXA_WHITE, CAIXA_ACCENT = "#00953B", "#003D1A", "#E8F5E9", "#4A4A4A", "#FFFFFF", "#FFB81C"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background: linear-gradient(135deg, #f5f5f5 0%, #fafafa 100%); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 24px; border-bottom: 3px solid {CAIXA_GREEN}; }}
.stTabs [aria-selected="true"] {{ color: {CAIXA_GREEN} !important; border-bottom: 3px solid {CAIXA_GREEN} !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background: linear-gradient(135deg, {CAIXA_GREEN} 0%, {CAIXA_DARK} 100%); padding: 40px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 4px 16px rgba(0, 149, 59, 0.3);">
    <h1 style="color: {CAIXA_WHITE}; margin: 0; font-size: 2.8rem; font-weight: 900;">🏦 Recomendador de Trilhas de IA</h1>
    <p style="color: {CAIXA_LIGHT}; margin: 10px 0 0 0; font-size: 1.1rem;">Caixa Econômica Federal | Sistema Inteligente de Recomendação</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("modelo_recomendacao_curso.pkl")

model = load_model()
classes = model.named_steps["model"].classes_

aba1, aba2, aba3, aba4 = st.tabs(["📋 Perfil", "🎯 Recomendação Premium", "📊 Modelo & Métricas", "🎓 Apresentação"])

with aba1:
    st.markdown(f"<h2 style='color: {CAIXA_GREEN};'>📋 Preencha o Perfil do Empregado</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("🏢 Área", ["Agencia Varejo", "Atendimento", "Controladoria", "Credito", "Financeiro", "Juridico", "Operacoes", "Prevencao a Fraudes", "Riscos", "TI"])
        funcao = st.selectbox("👔 Cargo", ["Analista", "Coordenador", "Desenvolvedor", "Especialista", "Gestor"])
        tempo_de_casa = st.slider("⏱️ Tempo (anos)", 0, 40, 5)
    with col2:
        ja_utilizou_ia = st.radio("🤖 Usou IA?", ["Sim", "Não"], horizontal=True)
        nivel_programacao = st.selectbox("💻 Programação", ["Nenhum", "Básico", "Intermediário", "Avançado"])
    forma_uso_ia = st.selectbox("🎯 Uso de IA", ["Ainda não sei", "Usuário negócio", "Ferramentas code/low-code", "Desenvolvedor", "Gestor/Líder"])
    impacto_erro_ia = st.selectbox("⚠️ Impacto", ["Ainda não sei", "Baixo", "Médio", "Alto"])
    atividade_principal = st.selectbox("📌 Atividade", ["Análise Governança", "Análise decisão", "Atendimento", "Gestão equipe", "Operação", "Relatórios", "Desenvolvimento"])
    objetivo_ia_6m = st.selectbox("🚀 Objetivo 6m", ["Explorar", "BI/Dashboards", "Automatizar", "Classificar", "Agentes IA", "Entender", "Prever", "Outro"])

    if st.button("💾 GERAR RECOMENDAÇÃO", type="primary", use_container_width=True):
        st.session_state["inp"] = pd.DataFrame([{"area": area, "funcao": funcao, "tempo_de_casa": tempo_de_casa, "ja_utilizou_ia": ja_utilizou_ia, "atividade_principal": atividade_principal, "objetivo_ia_6m": objetivo_ia_6m, "impacto_erro_ia": impacto_erro_ia, "forma_uso_ia": forma_uso_ia, "nivel_programacao": nivel_programacao}])
        st.success("✅ Perfil salvo! Vá para **🎯 Recomendação Premium**.")

with aba2:
    st.markdown(f"<h2 style='color: {CAIXA_GREEN};'>🎯 Trilhas Recomendadas</h2>", unsafe_allow_html=True)
    if "inp" not in st.session_state:
        st.info("👈 Preencha o perfil primeiro.")
    else:
        proba = model.predict_proba(st.session_state["inp"])[0]
        top_idx = np.argsort(-proba)[:3]
        medalhas, cores = ["🥇", "🥈", "🥉"], [f"linear-gradient(135deg, {CAIXA_GREEN} 0%, {CAIXA_DARK} 100%)", f"linear-gradient(135deg, {CAIXA_ACCENT} 0%, #FFA500 100%)", "linear-gradient(135deg, #CD7F32 0%, #8B4513 100%)"]

        for i, idx in enumerate(top_idx):
            curso, confianca = classes[idx], proba[idx] * 100
            st.markdown(f"""
            <div style="background: {cores[i]}; color: white; padding: 24px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15); border-left: 6px solid {['gold', 'silver', '#CD7F32'][i]};">
                <h3 style="margin: 0 0 12px 0; font-size: 1.4rem;">{medalhas[i]} {curso}</h3>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1; background: rgba(255,255,255,0.2); border-radius: 8px; height: 8px; margin-right: 12px;"><div style="background: white; height: 100%; width: {confianca}%;"></div></div>
                    <span style="font-size: 1.2rem; font-weight: bold;">{confianca:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_viz1, col_viz2 = st.columns([2, 1])
        with col_viz1:
            sorted_idx = np.argsort(-proba)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#f5f5f5")
            ax.set_facecolor("#ffffff")
            bars = ax.barh([classes[i][:20] for i in sorted_idx], [proba[i]*100 for i in sorted_idx], color=[CAIXA_GREEN if j < 3 else CAIXA_GRAY for j in range(len(sorted_idx))])
            ax.set_xlabel("Confiança (%)", color=CAIXA_GRAY, fontsize=11, fontweight="bold")
            ax.tick_params(colors=CAIXA_GRAY, labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(0, 100)
            for bar, val in zip(bars, [proba[i]*100 for i in sorted_idx]):
                ax.text(val + 1.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", color=CAIXA_GRAY, fontsize=9, fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with col_viz2:
            st.markdown(f"""
            <div style="background: {CAIXA_LIGHT}; padding: 16px; border-radius: 8px; border-left: 4px solid {CAIXA_GREEN};">
                <h4 style="color: {CAIXA_DARK}; margin-top: 0;">💡 Interpretação</h4>
                <p style="margin: 0; color: {CAIXA_GRAY}; font-size: 0.95rem;">Top 3 com maior alinhamento. Confiança baseada em 9 características.</p>
            </div>
            """, unsafe_allow_html=True)

with aba3:
    st.markdown(f"<h2 style='color: {CAIXA_GREEN};'>📊 Sobre o Modelo</h2>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div style="background: linear-gradient(135deg, {CAIXA_GREEN} 0%, {CAIXA_DARK} 100%); padding: 20px; border-radius: 12px; text-align: center; color: white;"><h1 style="margin: 0; font-size: 2.5rem;">92,8%</h1><p style="margin: 8px 0 0 0; font-size: 0.9rem;">Accuracy</p></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div style="background: linear-gradient(135deg, {CAIXA_ACCENT} 0%, #FFA500 100%); padding: 20px; border-radius: 12px; text-align: center; color: white;"><h1 style="margin: 0; font-size: 2.5rem;">92,4%</h1><p style="margin: 8px 0 0 0; font-size: 0.9rem;">F1-Score</p></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div style="background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%); padding: 20px; border-radius: 12px; text-align: center; color: white;"><h1 style="margin: 0; font-size: 2.5rem;">99,0%</h1><p style="margin: 8px 0 0 0; font-size: 0.9rem;">Top-3</p></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div style="background: linear-gradient(135deg, #558B2F 0%, #33691E 100%); padding: 20px; border-radius: 12px; text-align: center; color: white;"><h1 style="margin: 0; font-size: 2.5rem;">9,493</h1><p style="margin: 8px 0 0 0; font-size: 0.9rem;">Empregados</p></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<h3 style='color: {CAIXA_DARK};'>🧠 Arquitetura — Stacking (3 Bases + Meta)</h3>", unsafe_allow_html=True)
    st.markdown("| Camada | Algoritmo | Papel |
|---|---|---|
| Base 1 | Gradient Boosting | Padrões complexos |
| Base 2 | Random Forest | Robustez |
| Base 3 | Logistic Regression | Interpretabilidade |
| Meta | Logistic Regression | Combina predições |")

    st.divider()
    st.markdown(f"<h3 style='color: {CAIXA_DARK};'>📂 Dataset</h3>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    d1.metric("Empregados", "9.493")
    d2.metric("Features", "9")
    d3.metric("Cursos", "8")

with aba4:
    st.markdown(f"<h2 style='color: {CAIXA_GREEN};'>🎓 Para o Professor</h2>", unsafe_allow_html=True)
    st.markdown("""
### 📚 Resumo Executivo
Sistema de recomendação de trilhas IA com 92,8% acurácia.

### 🔍 Problema & Solução
| Aspecto | Detalhe |
|---|---|
| Problema | Empregados não sabem escolher trilha entre 8 cursos |
| Solução | ML Classifier (Stacking) personalizado por perfil |
| Dados | 9.493 empregados, 9 features |

### 📊 Metodologia
1. **EDA:** Análise de distribuição e correlações
2. **Pré-processamento:** OneHotEncoder + StandardScaler
3. **Modelagem:** Stacking (GB + RF + LR + LR meta)
4. **Validação:** StratifiedKFold 5-Fold

### 🎯 9 Features Utilizadas
1. Área atuação | 2. Função/Cargo | 3. Tempo casa | 4. Usou IA | 5. Atividade
6. Objetivo 6m | 7. Impacto erro | 8. Forma uso | 9. Nível programação

### 📈 Resultados
| Métrica | Valor |
|---|---|
| Accuracy | 92,8% |
| Macro F1 | 92,4% |
| Top-3 Accuracy | **99,0%** |

### 🚀 Deployment
- Frontend: Streamlit
- Backend: Scikit-learn
- Hosting: Streamlit Cloud
- Link: https://recomendador-cursos-caixa-vjct2rzvnnoki6c3dtgo6f.streamlit.app

### 💪 Impacto
✅ Facilita choice de 9.493+ empregados  
✅ Pronto para produção  
✅ Explainável e auditável
    """)
