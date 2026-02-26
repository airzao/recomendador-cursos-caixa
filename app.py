import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="Recomendador de Cursos — CAIXA", page_icon="🏦", layout="wide", initial_sidebar_state="collapsed")

CA_AZUL    = "#0070B8"
CA_ESCURO  = "#003F8A"
CA_LARANJA = "#F5A623"
CA_VERDE   = "#00A859"
CA_CINZA   = "#333333"
CA_CLARO   = "#F0F7FF"
CA_BRANCO  = "#FFFFFF"

# ── CSS: apenas fundo claro + inputs. SEM regra de cor geral em p/span. ──
st.markdown("""
<style>
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
section.main, .main {
    background-color: #F0F7FF !important;
    color: #333333 !important;
}
/* Labels dos inputs */
label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p {
    color: #333333 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background-color: #ffffff !important;
    color: #333333 !important;
    border: 1.5px solid #0070B8 !important;
    border-radius: 6px !important;
}
/* Radio */
[data-testid="stRadio"] label p { color: #333333 !important; }
/* Slider */
[data-testid="stSlider"] p { color: #333333 !important; }
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 3px solid #0070B8 !important;
    background-color: #F0F7FF !important;
}
.stTabs [data-baseweb="tab"] {
    color: #333333 !important;
    background-color: #F0F7FF !important;
    font-weight: 600;
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #0070B8 !important;
    background-color: #ffffff !important;
    border-bottom: 3px solid #0070B8 !important;
}
/* Tabelas markdown */
thead tr th { background-color: #0070B8 !important; color: white !important; padding: 8px !important; }
tbody tr td { color: #333333 !important; padding: 8px !important; }
tbody tr:nth-child(even) td { background-color: #E8F4FF !important; }
/* Métricas nativas */
[data-testid="metric-container"] { background: white; border-radius: 8px; padding: 12px; border: 1px solid #dee2e6; }
[data-testid="metric-container"] label { color: #555555 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #003F8A !important; font-weight: 900 !important; }
/* Botão */
[data-testid="baseButton-primary"] {
    background-color: #0070B8 !important; color: white !important;
    font-weight: 700 !important; border-radius: 8px !important; border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# HEADER — fonte branca explícita em tudo
# ═══════════════════════════════════════════
st.markdown(f"""
<div style="background:linear-gradient(135deg,{CA_AZUL} 0%,{CA_ESCURO} 100%);
            padding:36px 40px;border-radius:12px;margin-bottom:28px;
            box-shadow:0 6px 24px rgba(0,63,138,0.25);">
  <h1 style="color:#ffffff !important;margin:0;font-size:2.4rem;font-weight:900;letter-spacing:-0.5px;">
    🏦 Recomendador de Trilhas de IA
  </h1>
  <p style="color:#ffffff !important;margin:8px 0 0 0;font-size:1.05rem;opacity:0.9;">
    Caixa Econômica Federal &nbsp;|&nbsp; Sistema Inteligente de Recomendação &nbsp;|&nbsp;
    <span style="background:{CA_LARANJA};color:#ffffff !important;padding:2px 10px;border-radius:12px;font-size:0.85rem;font-weight:700;">ML • Stacking Classifier</span>
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("modelo_recomendacao_curso.pkl")

model = load_model()
classes = model.named_steps["model"].classes_

aba1, aba2, aba3, aba4 = st.tabs(["📋 Perfil", "🎯 Recomendação Premium", "📊 Modelo & Métricas", "🎓 Para o Professor"])

# ══════════════════════════════════════════════════════════════════════
# ABA 1
# ══════════════════════════════════════════════════════════════════════
with aba1:
    st.markdown(f"<h2 style='color:{CA_AZUL};margin-bottom:4px;'>📋 Perfil do Empregado</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CA_CINZA};margin-top:0;margin-bottom:20px;'>Preencha os dados abaixo para receber recomendações personalizadas.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("🏢 Área de atuação", ["Agencia Varejo","Atendimento","Controladoria","Credito","Financeiro","Juridico","Operacoes","Prevencao a Fraudes","Riscos","TI"])
        funcao = st.selectbox("👔 Função / Cargo", ["Analista","Coordenador","Desenvolvedor","Especialista","Gestor"])
        tempo_de_casa = st.slider("⏱️ Tempo de casa (anos)", 0, 40, 5)
    with col2:
        ja_utilizou_ia = st.radio("🤖 Já utilizou alguma IA?", ["Sim","Nao"], horizontal=True)
        nivel_programacao = st.selectbox("💻 Nível de programação", ["Nenhum","Basico leio ajusto scripts simples","Intermediario desenvolvo scripts aplicacoes com autonomia","Avancado integracoes debugging boas praticas"])

    col3, col4 = st.columns(2)
    with col3:
        forma_uso_ia = st.selectbox("🎯 Como pretende usar IA?", ["Ainda nao sei","Como usuario a de negocio sem programar","Como usuario a com ferramentas no code low code","Como desenvolvedor a programando integracoes solucoes","Como gestor a lider definindo prioridades e direcionando uso"])
        impacto_erro_ia = st.selectbox("⚠️ Impacto se a IA errar", ["Ainda nao sei","Baixo retrabalho pequeno ajuste simples","Medio atraso retrabalho relevante comunicacao incorreta","Alto risco financeiro juridico compliance reputacao decisao"])
    with col4:
        atividade_principal = st.selectbox("📌 Atividade principal", ["Analise de Governaca riscos compliance e controles","Analise para apoio a decisao indicadores desempenho","Atendimento a demandas clientes fornecedores areas internas","Gestao e priorizacao planejamento coordenacao de equipe","Operacao e rotinas administrativas cadastro conferencia","Producao e consolidacao de relatorios apresentacoes","Desenvolvimento de sistemas integracoes APIs"])
        objetivo_ia_6m = st.selectbox("🚀 Objetivo com IA nos próximos 6 meses", ["Ainda estou explorando quero entender possibilidades","Apoiar decisoes com dashboards BI IA","Automatizar tarefas e fluxos","Classificar organizar informacoes ex documentos tickets","Criar assistentes agentes de IA para apoiar equipes","Entender como a IA funciona","Prever resultados ex demanda risco churn fraude desempenho","Outro"])

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 GERAR RECOMENDAÇÃO PERSONALIZADA", type="primary", use_container_width=True):
        st.session_state["inp"] = pd.DataFrame([{
            "area":area,"funcao":funcao,"tempo_de_casa":tempo_de_casa,
            "ja_utilizou_ia":ja_utilizou_ia,"atividade_principal":atividade_principal,
            "objetivo_ia_6m":objetivo_ia_6m,"impacto_erro_ia":impacto_erro_ia,
            "forma_uso_ia":forma_uso_ia,"nivel_programacao":nivel_programacao
        }])
        st.success("✅ Perfil salvo! Vá para a aba **🎯 Recomendação Premium**.")

# ══════════════════════════════════════════════════════════════════════
# ABA 2
# ══════════════════════════════════════════════════════════════════════
with aba2:
    st.markdown(f"<h2 style='color:{CA_AZUL};'>🎯 Trilhas Recomendadas para o seu Perfil</h2>", unsafe_allow_html=True)
    if "inp" not in st.session_state:
        st.info("👈 Preencha o perfil na aba **📋 Perfil** primeiro.")
    else:
        proba    = model.predict_proba(st.session_state["inp"])[0]
        top_idx  = np.argsort(-proba)[:3]
        medalhas = ["🥇","🥈","🥉"]
        fundos   = [
            f"linear-gradient(135deg,{CA_AZUL} 0%,{CA_ESCURO} 100%)",
            f"linear-gradient(135deg,{CA_LARANJA} 0%,#E08B00 100%)",
            f"linear-gradient(135deg,#4A7C59 0%,#2D5A3D 100%)"
        ]
        bordas = [CA_LARANJA, "#ffffff", CA_CLARO]

        st.markdown(f"<h3 style='color:{CA_ESCURO};margin-bottom:16px;'>TOP 3 Cursos Recomendados</h3>", unsafe_allow_html=True)

        for i, idx in enumerate(top_idx):
            curso = classes[idx]
            conf  = proba[idx] * 100
            # — tudo com color:#ffffff !important inline —
            st.markdown(f"""
<div style="background:{fundos[i]};padding:28px 32px;border-radius:14px;
            margin-bottom:14px;box-shadow:0 8px 24px rgba(0,63,138,0.18);
            border-left:6px solid {bordas[i]};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <h3 style="margin:0;font-size:1.35rem;font-weight:800;color:#ffffff !important;">{medalhas[i]} {curso}</h3>
    <span style="background:rgba(255,255,255,0.25);color:#ffffff !important;
                 padding:4px 16px;border-radius:20px;font-size:1rem;font-weight:700;">{conf:.1f}%</span>
  </div>
  <div style="background:rgba(255,255,255,0.2);border-radius:8px;height:10px;overflow:hidden;">
    <div style="background:#ffffff;height:100%;width:{conf:.0f}%;border-radius:8px;"></div>
  </div>
  <p style="margin:10px 0 0 0;font-size:0.85rem;color:#ffffff !important;opacity:0.85;">
    Confiança do modelo para este perfil
  </p>
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"<h3 style='color:{CA_ESCURO};'>📊 Probabilidade — Todos os Cursos</h3>", unsafe_allow_html=True)

        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            sorted_idx = np.argsort(-proba)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#F0F7FF"); ax.set_facecolor("#ffffff")
            cores = [CA_AZUL if j < 3 else "#AAAAAA" for j in range(len(sorted_idx))]
            ax.barh([classes[i] for i in sorted_idx], [proba[i]*100 for i in sorted_idx], color=cores, height=0.6)
            ax.set_xlabel("Confiança (%)", color=CA_CINZA, fontsize=11, fontweight="bold")
            ax.tick_params(colors=CA_CINZA, labelsize=10); ax.set_xlim(0, 105)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#dddddd"); ax.spines["bottom"].set_color("#dddddd")
            for i_, idx_ in enumerate(sorted_idx):
                ax.text(proba[idx_]*100+1.5, i_, f"{proba[idx_]*100:.1f}%", va="center", color=CA_CINZA, fontsize=9, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        with col_v2:
            perfil = st.session_state["inp"].iloc[0]
            st.markdown(f"""
<div style="background:white;padding:20px;border-radius:10px;
            border-left:5px solid {CA_AZUL};box-shadow:0 2px 8px rgba(0,112,184,0.1);">
  <h4 style="color:{CA_AZUL} !important;margin-top:0;margin-bottom:12px;">👤 Perfil Analisado</h4>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Área:</b> {perfil['area']}</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Cargo:</b> {perfil['funcao']}</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Casa:</b> {perfil['tempo_de_casa']} anos</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Já usou IA:</b> {perfil['ja_utilizou_ia']}</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Programação:</b> {perfil['nivel_programacao']}</p>
  <hr style="border-color:#0070B8;opacity:0.2;margin:12px 0;">
  <p style="margin:0;color:{CA_AZUL} !important;font-size:0.8rem;font-style:italic;">9 features analisadas pelo modelo</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ABA 3
# ══════════════════════════════════════════════════════════════════════
with aba3:
    st.markdown(f"<h2 style='color:{CA_AZUL};'>📊 Modelo & Métricas</h2>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    kpis = [
        (m1, "92,8%", "Accuracy",      CA_AZUL,   CA_ESCURO),
        (m2, "92,4%", "Macro F1",      CA_LARANJA,"#E08B00"),
        (m3, "99,0%", "Top-3 Accuracy",CA_VERDE,  "#007A40"),
        (m4, "9.493", "Empregados",    "#7B2D8B", "#4A1A55"),
    ]
    for col_, val_, lbl_, c1_, c2_ in kpis:
        with col_:
            # h1 e p com color:#ffffff !important
            st.markdown(f"""
<div style="background:linear-gradient(135deg,{c1_} 0%,{c2_} 100%);
            padding:22px;border-radius:12px;text-align:center;
            box-shadow:0 4px 16px rgba(0,0,0,0.12);margin-bottom:8px;">
  <h1 style="margin:0;font-size:2.2rem;font-weight:900;color:#ffffff !important;">{val_}</h1>
  <p style="margin:8px 0 0 0;font-size:0.9rem;font-weight:600;color:#ffffff !important;opacity:0.9;">{lbl_}</p>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<h3 style='color:{CA_ESCURO};'>🧠 Arquitetura — Stacking Classifier</h3>", unsafe_allow_html=True)
    arch1, arch2 = st.columns([1.6, 1])
    with arch1:
        st.markdown("""
| Camada | Algoritmo | Parâmetros | Papel |
|--------|-----------|------------|-------|
| **Base 1** | Gradient Boosting | n_estimators=150 | Padrões complexos e não-lineares |
| **Base 2** | Random Forest | n_estimators=100 | Robustez e generalização |
| **Base 3** | Logistic Regression | max_iter=1000 | Fronteiras lineares |
| **Meta** | Logistic Regression | cv=5 | Combina as 3 predições |
        """)
        st.markdown(f"""
<div style="background:{CA_CLARO};padding:14px 18px;border-radius:8px;
            border-left:5px solid {CA_AZUL};margin-top:12px;">
  <p style="margin:0;color:{CA_CINZA} !important;font-size:0.92rem;">
    <b style="color:{CA_AZUL} !important;">Por que Stacking?</b>
    Cada algoritmo captura um aspecto diferente dos dados.
    O meta-modelo aprende a ponderar as três predições,
    resultando em acurácia superior a qualquer modelo isolado.
  </p>
</div>""", unsafe_allow_html=True)

    with arch2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
        fig2.patch.set_facecolor("#F0F7FF"); ax2.set_facecolor("#F0F7FF")
        ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
        bxs = [
            (0.2,7.2,3.5,1.3,CA_AZUL,   "Gradient Boosting"),
            (0.2,4.7,3.5,1.3,CA_LARANJA,"Random Forest"),
            (0.2,2.2,3.5,1.3,"#666666", "Logistic Regression"),
            (5.3,3.8,4.2,2.0,CA_ESCURO, "Meta\nLogistic\nRegression"),
        ]
        for x,y,w,h,cor,lbl in bxs:
            ax2.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15",facecolor=cor,edgecolor="white",linewidth=2.5))
            ax2.text(x+w/2,y+h/2,lbl,ha="center",va="center",color="white",fontsize=9.5,fontweight="bold")
        for y_s in [7.85,5.35,2.85]:
            ax2.annotate("",xy=(5.3,4.8),xytext=(3.7,y_s),arrowprops=dict(arrowstyle="->",color=CA_AZUL,lw=2.5))
        ax2.annotate("",xy=(9.8,4.8),xytext=(9.5,4.8),arrowprops=dict(arrowstyle="->",color=CA_LARANJA,lw=3.5))
        ax2.text(9.85,4.8,"🎯",fontsize=18,va="center")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.divider()
    st.markdown(f"<h3 style='color:{CA_ESCURO};'>📂 Dataset</h3>", unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("👥 Empregados","9.493"); d2.metric("📊 Features","9")
    d3.metric("🎓 Cursos","8"); d4.metric("✅ Completude","100%")

    cursos_l = ["Fundamentos IA","IA Explicável","Automação","IA Negócios","Prompting","Agentes IA","RAG","ML Negócio"]
    qtds_l   = [2375,1768,1231,1129,904,871,774,441]
    fig3, ax3 = plt.subplots(figsize=(10, 3.8))
    fig3.patch.set_facecolor("#F0F7FF"); ax3.set_facecolor("#ffffff")
    ax3.bar(range(len(cursos_l)),qtds_l,color=CA_AZUL,alpha=0.85,edgecolor=CA_ESCURO,linewidth=1.5,width=0.6)
    ax3.set_ylabel("Quantidade",color=CA_CINZA,fontweight="bold")
    ax3.set_xticks(range(len(cursos_l)))
    ax3.set_xticklabels(cursos_l,rotation=30,ha="right",fontsize=9,color=CA_CINZA)
    ax3.tick_params(colors=CA_CINZA)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("#dddddd"); ax3.spines["bottom"].set_color("#dddddd")
    for i,v in enumerate(qtds_l):
        ax3.text(i,v+50,str(v),ha="center",va="bottom",color=CA_ESCURO,fontsize=8,fontweight="bold")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.divider()
    st.markdown(f"<h3 style='color:{CA_ESCURO};'>⚙️ Decisões Técnicas</h3>", unsafe_allow_html=True)
    t1,t2 = st.columns(2)
    with t1:
        st.markdown(f"""
<div style="background:white;padding:16px;border-radius:8px;
            border-left:4px solid {CA_AZUL};box-shadow:0 2px 8px rgba(0,112,184,0.08);">
  <p style="color:{CA_CINZA} !important;margin:4px 0;"><b>🔄 Desbalanceamento:</b> class_weight='balanced'</p>
  <p style="color:{CA_CINZA} !important;margin:4px 0;"><b>🔧 Pipeline:</b> OneHotEncoder + StandardScaler</p>
  <p style="color:{CA_CINZA} !important;margin:4px 0;"><b>🛡️ Data Leakage:</b> fit() só no X_train</p>
</div>""", unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
<div style="background:white;padding:16px;border-radius:8px;
            border-left:4px solid {CA_LARANJA};box-shadow:0 2px 8px rgba(245,166,35,0.08);">
  <p style="color:{CA_CINZA} !important;margin:4px 0;"><b>✅ Validação:</b> StratifiedKFold (5 folds)</p>
  <p style="color:{CA_CINZA} !important;margin:4px 0;"><b>📊 Métrica:</b> Macro F1-Score</p>
  <p style="color:{CA_CINZA} !important;margin:4px 0;"><b>💾 Modelo:</b> joblib compress=3 (&lt;10 MB)</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ABA 4
# ══════════════════════════════════════════════════════════════════════
with aba4:
    st.markdown(f"<h2 style='color:{CA_AZUL};'>🎓 Documentação para Apresentação</h2>", unsafe_allow_html=True)

    # Resumo executivo — TODO texto com color:#ffffff !important
    st.markdown(f"""
<div style="background:linear-gradient(135deg,{CA_AZUL} 0%,{CA_ESCURO} 100%);
            padding:28px 32px;border-radius:14px;margin-bottom:24px;">
  <h3 style="margin:0 0 12px 0;font-size:1.4rem;color:#ffffff !important;">📚 Resumo Executivo</h3>
  <p style="margin:0;font-size:1rem;line-height:1.7;color:#ffffff !important;">
    Sistema inteligente de recomendação de trilhas de Inteligência Artificial
    para os empregados da
    <span style="color:#ffffff !important;font-weight:700;text-decoration:underline;">Caixa Econômica Federal</span>,
    desenvolvido com
    <span style="color:#ffffff !important;font-weight:700;">Machine Learning supervisionado</span>.
    O modelo atinge
    <span style="color:#ffffff !important;font-weight:700;">92,8% de acurácia</span>
    e garante que o curso ideal esteja entre os 3 recomendados em
    <span style="color:#ffffff !important;font-weight:700;">99% dos casos</span>.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### 🔍 Problema & Solução

| Aspecto | Detalhe |
|---|---|
| **Problema** | 9.493 empregados sem direcionamento claro entre 8 trilhas de IA |
| **Solução** | Classificador ML personalizado por 9 features do perfil profissional |
| **Abordagem** | Supervised Learning — Multiclass Classification |
| **Resultado** | Top-3 Accuracy de 99% — curso ideal sempre entre os recomendados |

---

### 📊 Metodologia Completa

1. **EDA** — Análise de distribuição das classes, correlações entre features e target
2. **Feature Engineering** — 9 variáveis categóricas e numéricas selecionadas
3. **Pré-processamento** — `Pipeline` com `OneHotEncoder` + `StandardScaler`
4. **Tratamento de desbalanceamento** — `class_weight='balanced'`
5. **Modelagem** — `StackingClassifier`: GB + RF + LR (bases) → LR (meta)
6. **Validação** — `StratifiedKFold` 5-Fold com Macro F1-Score

---

### 🎯 As 9 Features do Modelo

| # | Feature | Tipo | Categorias |
|---|---------|------|------------|
| 1 | Área de atuação | Categórica | 10 |
| 2 | Função/Cargo | Categórica | 5 |
| 3 | Tempo de casa | Numérica | — |
| 4 | Já utilizou IA | Binária | 2 |
| 5 | Atividade principal | Categórica | 7 |
| 6 | Objetivo 6 meses | Categórica | 8 |
| 7 | Impacto do erro | Categórica | 4 |
| 8 | Forma de uso de IA | Categórica | 5 |
| 9 | Nível de programação | Categórica | 4 |

---

### 📈 Resultados no Test Set

| Métrica | Valor | Interpretação |
|---|---|---|
| **Accuracy** | 92,8% | Acertos exatos |
| **Macro F1-Score** | 92,4% | Média entre todas as classes |
| **Weighted F1** | 92,9% | Ponderado pelo tamanho da classe |
| **Top-3 Accuracy** | **99,0%** | Curso ideal entre os 3 recomendados |

---

### 🚀 Deployment & Tecnologias

- **Linguagem:** Python 3.13
- **ML:** Scikit-learn 1.6.1 (StackingClassifier)
- **Frontend:** Streamlit 1.54
- **Versionamento:** GitHub
- **Hosting:** Streamlit Cloud (CI/CD automático)
    """)
