import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Recomendador de Cursos — CAIXA", 
    page_icon="🏦", 
    layout="wide", 
    initial_sidebar_state="expanded" # Sidebar aberta para mostrar a simulação
)

CA_AZUL    = "#0070B8"
CA_ESCURO  = "#003F8A"
CA_LARANJA = "#F5A623"
CA_VERDE   = "#00A859"
CA_CINZA   = "#333333"
CA_CLARO   = "#F0F7FF"
CA_BRANCO  = "#FFFFFF"

# ═══════════════════════════════════════════
# CSS GLOBAL E DA SIDEBAR
# ═══════════════════════════════════════════
st.markdown("""
<style>
html, body,[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="block-container"], section.main, .main {
    background-color: #F0F7FF !important;
    color: #333333 !important;
}
.banner-white, .banner-white h1, .banner-white h2, .banner-white h3, .banner-white h4, .banner-white p, .banner-white span, .banner-white b, .banner-white strong {
    color: #ffffff !important;
}
label,[data-testid="stWidgetLabel"] p {
    color: #333333 !important; font-weight: 600 !important; font-size: 0.95rem !important;
}
[data-testid="stSelectbox"] > div > div { background-color: #ffffff !important; border: 1.5px solid #0070B8 !important; border-radius: 6px !important; }
.stTabs [data-baseweb="tab-list"] { border-bottom: 3px solid #0070B8 !important; background-color: #F0F7FF !important; }
.stTabs[data-baseweb="tab"] { color: #333333 !important; font-weight: 600; padding: 10px 20px; }
.stTabs [aria-selected="true"] { color: #0070B8 !important; background-color: #ffffff !important; border-bottom: 3px solid #0070B8 !important; }
[data-testid="metric-container"],[data-testid="stMetric"] { background-color: #ffffff !important; border-radius: 10px !important; padding: 16px !important; border: 1px solid #dee2e6 !important; box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important; }
[data-testid="stMetricLabel"] *,[data-testid="metric-container"] label { color: #555555 !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] *, [data-testid="stMetricValue"] { color: #003F8A !important; font-weight: 900 !important; }
[data-testid="baseButton-primary"] { background-color: #0070B8 !important; color: white !important; font-weight: 700 !important; border-radius: 8px !important; border: none !important; }
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #dee2e6; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# LOAD DOS 3 MODELOS
# ═══════════════════════════════════════════
@st.cache_resource
def load_models():
    # Carrega o dicionário salvo no Passo 1
    return joblib.load("modelos_comparacao.pkl")

modelos_dict = load_models()

# ═══════════════════════════════════════════
# SIDEBAR - SIMULADOR DINÂMICO
# ═══════════════════════════════════════════
st.sidebar.markdown(f"<h2 style='color:{CA_AZUL};'>🛠️ Simulador de Modelos</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='color:{CA_CINZA}; font-size:0.9rem;'>Escolha o algoritmo abaixo para ver como o aplicativo e as métricas reagem em tempo real.</p>", unsafe_allow_html=True)

modelo_selecionado = st.sidebar.radio(
    "Modelo Ativo:",
    ["Gradient Boosting 🏆", "Random Forest", "Regressão Logística"]
)

# Mapeamento do modelo real baseado na seleção
if "Gradient Boosting" in modelo_selecionado:
    model_ativo = modelos_dict["GB"]
    nome_tecnico = "Gradient Boosting"
    cor_tema = CA_AZUL
elif "Random Forest" in modelo_selecionado:
    model_ativo = modelos_dict["RF"]
    nome_tecnico = "Random Forest"
    cor_tema = CA_VERDE
else:
    model_ativo = modelos_dict["LR"]
    nome_tecnico = "Regressão Logística"
    cor_tema = CA_CINZA

classes = model_ativo.named_steps["model"].classes_

# Dicionário dinâmico de métricas (números reais tirados do seu Notebook!)
kpis_dinamicos = {
    "Gradient Boosting 🏆": {"acc": "93,4%", "f1": "92,9%", "top3": "97,7%", "desc": "Melhor performance geral. Captura relações complexas nos dados e corrige vieses sequencialmente."},
    "Random Forest":        {"acc": "91,7%", "f1": "91,3%", "top3": "96,5%", "desc": "Modelo robusto baseado em múltiplas árvores, mas que apresentou leve queda na métrica Macro F1."},
    "Regressão Logística":  {"acc": "88,5%", "f1": "87,4%", "top3": "94,2%", "desc": "Modelo baseline simples. Funciona bem como base, mas sofre para entender perfis mais complexos."}
}

# ═══════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════
st.markdown(f"""
<div class="banner-white" 
     style="background:linear-gradient(135deg,{cor_tema} 0%,{CA_ESCURO} 100%);
            padding:36px 40px;border-radius:12px;margin-bottom:28px;
            box-shadow:0 6px 24px rgba(0,0,0,0.25);">
  <h1 style="margin:0;font-size:2.4rem;font-weight:900;letter-spacing:-0.5px;">
    🏦 Recomendador de Trilhas de IA
  </h1>
  <p style="margin:8px 0 0 0;font-size:1.05rem;opacity:0.9;">
    Caixa Econômica Federal &nbsp;|&nbsp; Sistema Inteligente de Recomendação &nbsp;|&nbsp;
    <span style="background:{CA_LARANJA};padding:2px 10px;border-radius:12px;font-size:0.85rem;font-weight:700;">{nome_tecnico}</span>
  </p>
</div>
""", unsafe_allow_html=True)

aba1, aba2, aba3, aba4 = st.tabs(["📋 Perfil", "🎯 Recomendação Premium", "📊 Modelo & Métricas", "🎓 Para o Professor"])

# ══════════════════════════════════════════════════════════════════════
# ABA 1 — PERFIL
# ══════════════════════════════════════════════════════════════════════
with aba1:
    st.markdown(f"<h2 style='color:{CA_AZUL};margin-bottom:4px;'>📋 Perfil do Empregado</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CA_CINZA};margin-top:0;margin-bottom:20px;'>Preencha os dados abaixo para receber recomendações personalizadas.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("🏢 Área de atuação",["Agencia Varejo","Atendimento","Controladoria","Credito","Financeiro","Juridico","Operacoes","Prevencao a Fraudes","Riscos","TI","Dados e Analytics","PMO","RH","Produtos","Auditoria Interna","Compliance","Seguranca da Informacao"])
        funcao = st.selectbox("👔 Função / Cargo",["Analista","Coordenador","Desenvolvedor","Especialista","Gestor"])
        tempo_de_casa = st.slider("⏱️ Tempo de casa (anos)", 0, 40, 5)
    with col2:
        ja_utilizou_ia = st.radio("🤖 Já utilizou alguma IA?",["Sim","Nao"], horizontal=True)
        nivel_programacao = st.selectbox("💻 Nível de programação",["Nenhum","Basico leio ajusto scripts simples","Intermediario desenvolvo scripts aplicacoes com autonomia","Avancado integracoes debugging boas praticas"])

    col3, col4 = st.columns(2)
    with col3:
        forma_uso_ia = st.selectbox("🎯 Como pretende usar IA?",["Ainda nao sei","Como usuario a de negocio sem programar","Como usuario a com ferramentas no code low code","Como desenvolvedor a programando integracoes solucoes","Como gestor a lider definindo prioridades e direcionando uso"])
        impacto_erro_ia = st.selectbox("⚠️ Impacto se a IA errar",["Ainda nao sei","Baixo retrabalho pequeno ajuste simples","Medio atraso retrabalho relevante comunicacao incorreta","Alto risco financeiro juridico compliance reputacao decisao"])
    with col4:
        atividade_principal = st.selectbox("📌 Atividade principal",["Analise de Governanca riscos compliance e controles","Analise para apoio a decisao indicadores desempenho relatorios analiticos","Atendimento a demandas clientes fornecedores areas internas","Gestao e priorizacao planejamento coordenacao de equipe definicao de prioridades","Operacao e rotinas administrativas cadastro conferencia execucao de processos","Producao e consolidacao de relatorios apresentacoes","Desenvolvimento tecnico scripts integracoes sistemas","Outro"])
        objetivo_ia_6m = st.selectbox("🚀 Objetivo com IA nos próximos 6 meses",["Ainda estou explorando quero entender possibilidades","Apoiar decisoes com dashboards BI IA","Automatizar tarefas e fluxos","Classificar organizar informacoes ex documentos tickets e mails textos","Criar assistentes agentes de IA para apoiar equipes ou processos","Entender como a IA funciona","Prever resultados ex demanda risco churn fraude desempenho usando modelos","Buscar informacao em bases internas ex FAQ documentos knowledge base RAG","Analisar dados e gerar insights com mais velocidade","Melhorar controles auditoria risco ou compliance com IA","Gerar conteudo textos resumos apresentacoes documentacao","Outro"])

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
# ABA 2 — RECOMENDAÇÃO DINÂMICA
# ══════════════════════════════════════════════════════════════════════
with aba2:
    st.markdown(f"<h2 style='color:{CA_AZUL};'>🎯 Trilhas Recomendadas ({nome_tecnico})</h2>", unsafe_allow_html=True)
    if "inp" not in st.session_state:
        st.info("👈 Preencha o perfil na aba **📋 Perfil** primeiro.")
    else:
        proba    = model_ativo.predict_proba(st.session_state["inp"])[0]
        top_idx  = np.argsort(-proba)[:3]
        medalhas = ["🥇","🥈","🥉"]
        fundos   =[
            f"linear-gradient(135deg,{cor_tema} 0%,{CA_ESCURO} 100%)",
            f"linear-gradient(135deg,{CA_LARANJA} 0%,#E08B00 100%)",
            f"linear-gradient(135deg,#4A7C59 0%,#2D5A3D 100%)"
        ]
        bordas =[CA_LARANJA, "#ffffff", CA_CLARO]

        st.markdown(f"<h3 style='color:{CA_ESCURO};margin-bottom:16px;'>TOP 3 Cursos Recomendados</h3>", unsafe_allow_html=True)

        for i, idx in enumerate(top_idx):
            curso = classes[idx]
            conf  = proba[idx] * 100
            
            st.markdown(f"""
<div class="banner-white" 
     style="background:{fundos[i]};padding:28px 32px;border-radius:14px;
            margin-bottom:14px;box-shadow:0 8px 24px rgba(0,0,0,0.18);
            border-left:6px solid {bordas[i]};">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <h3 style="margin:0;font-size:1.35rem;font-weight:800;">{medalhas[i]} {curso}</h3>
    <span style="background:rgba(255,255,255,0.25);padding:4px 16px;border-radius:20px;font-size:1rem;font-weight:700;">{conf:.1f}%</span>
  </div>
  <div style="background:rgba(255,255,255,0.2);border-radius:8px;height:10px;overflow:hidden;">
    <div style="background:#ffffff;height:100%;width:{conf:.0f}%;border-radius:8px;"></div>
  </div>
  <p style="margin:10px 0 0 0;font-size:0.85rem;opacity:0.85;">
    Confiança do {nome_tecnico} para este perfil
  </p>
</div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown(f"<h3 style='color:{CA_ESCURO};'>📊 Probabilidade — Todos os Cursos</h3>", unsafe_allow_html=True)

        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            sorted_idx = np.argsort(-proba)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#F0F7FF"); ax.set_facecolor("#ffffff")
            cores =[cor_tema if j < 3 else "#AAAAAA" for j in range(len(sorted_idx))]
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
            border-left:5px solid {cor_tema};box-shadow:0 2px 8px rgba(0,0,0,0.1);">
  <h4 style="color:{cor_tema} !important;margin-top:0;margin-bottom:12px;">👤 Perfil Analisado</h4>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Área:</b> {perfil['area']}</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Cargo:</b> {perfil['funcao']}</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Casa:</b> {perfil['tempo_de_casa']} anos</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Já usou IA:</b> {perfil['ja_utilizou_ia']}</p>
  <p style="margin:4px 0;color:{CA_CINZA} !important;font-size:0.88rem;"><b>Programação:</b> {perfil['nivel_programacao']}</p>
  <hr style="border-color:#dddddd;margin:12px 0;">
  <p style="margin:0;color:{cor_tema} !important;font-size:0.8rem;font-style:italic;">9 features avaliadas</p>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ABA 3 — MODELO & MÉTRICAS DINÂMICAS
# ══════════════════════════════════════════════════════════════════════
with aba3:
    dados_kpi = kpis_dinamicos[modelo_selecionado]
    
    st.markdown(f"<h2 style='color:{cor_tema};'>📊 Desempenho no Test Set ({nome_tecnico})</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CA_CINZA};'><b>Comportamento do algoritmo:</b> {dados_kpi['desc']}</p>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    kpis =[
        (m1, dados_kpi["acc"], "Accuracy",      cor_tema,   CA_ESCURO),
        (m2, dados_kpi["f1"], "Macro F1",      CA_LARANJA,"#E08B00"),
        (m3, dados_kpi["top3"], "Top-3 Accuracy",CA_VERDE,  "#007A40"),
        (m4, "9.493", "Total Empregados",    "#7B2D8B", "#4A1A55"),
    ]
    
    for col_, val_, lbl_, c1_, c2_ in kpis:
        with col_:
            st.markdown(f"""
<div class="banner-white" 
     style="background:linear-gradient(135deg,{c1_} 0%,{c2_} 100%);
            padding:22px;border-radius:12px;text-align:center;
            box-shadow:0 4px 16px rgba(0,0,0,0.12);margin-bottom:8px;">
  <h1 style="margin:0;font-size:2.2rem;font-weight:900;">{val_}</h1>
  <p style="margin:8px 0 0 0;font-size:0.9rem;font-weight:600;opacity:0.9;">{lbl_}</p>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<h3 style='color:{CA_ESCURO};'>📂 Dados do Treinamento</h3>", unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("👥 Amostras Treino","7.594"); d2.metric("👥 Amostras Teste","1.899")
    d3.metric("📊 Features Avaliadas","9"); d4.metric("🎓 Total de Cursos","8")

    cursos_l =["Fundamentos IA","IA Explicável","Automação","IA Negócios","Prompting","Agentes IA","RAG","ML Negócio"]
    qtds_l   =[2375,1768,1231,1129,904,871,774,441]
    fig3, ax3 = plt.subplots(figsize=(10, 3.8))
    fig3.patch.set_facecolor("#F0F7FF"); ax3.set_facecolor("#ffffff")
    ax3.bar(range(len(cursos_l)),qtds_l,color=cor_tema,alpha=0.85,edgecolor=CA_ESCURO,linewidth=1.5,width=0.6)
    ax3.set_ylabel("Quantidade",color=CA_CINZA,fontweight="bold")
    ax3.set_xticks(range(len(cursos_l)))
    ax3.set_xticklabels(cursos_l,rotation=30,ha="right",fontsize=9,color=CA_CINZA)
    ax3.tick_params(colors=CA_CINZA)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_color("#dddddd"); ax3.spines["bottom"].set_color("#dddddd")
    for i,v in enumerate(qtds_l):
        ax3.text(i,v+50,str(v),ha="center",va="bottom",color=CA_ESCURO,fontsize=8,fontweight="bold")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# ══════════════════════════════════════════════════════════════════════
# ABA 4 — PARA O PROFESSOR (TEXTO DINÂMICO)
# ══════════════════════════════════════════════════════════════════════
with aba4:
    st.markdown(f"<h2 style='color:{cor_tema};'>🎓 Documentação para Apresentação</h2>", unsafe_allow_html=True)

    st.markdown(f"""
<div class="banner-white" 
     style="background:linear-gradient(135deg,{cor_tema} 0%,{CA_ESCURO} 100%);
            padding:28px 32px;border-radius:14px;margin-bottom:24px;">
  <h3 style="margin:0 0 12px 0;font-size:1.4rem;">📚 Resumo Executivo</h3>
  <p style="margin:0;font-size:1rem;line-height:1.7;">
    Sistema inteligente de recomendação de trilhas de Inteligência Artificial
    para os empregados da
    <span style="font-weight:700;text-decoration:underline;">Caixa Econômica Federal</span>.
    A prova de conceito (PoC) compara três algoritmos: Regressão Logística, Random Forest e Gradient Boosting.
    O vencedor escolhido para produção foi o <span style="font-weight:700;">Gradient Boosting Classifier</span> 
    que atingiu <span style="font-weight:700;">93,4% de acurácia</span> 
    e garante que o curso ideal esteja entre os 3 recomendados em 
    <span style="font-weight:700;">97,7% dos casos</span>.
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### 🔍 O Problema & A Solução

| Aspecto | Detalhe |
|---|---|
| **Problema** | 9.493 empregados sem direcionamento claro entre 8 trilhas complexas de IA |
| **Solução** | Construir um Sistema de Recomendação baseado em Machine Learning (Classificação Multiclasse) |
| **Trade-offs Identificados** | A regressão logística subaproveita os dados. O Random Forest sobre com underfitting/tamanho de arquivo, sendo o Gradient Boosting o mais equilibrado em processamento vs. acurácia. |

---

### 📊 Metodologia e Pipeline (Idêntico ao Jupyter Notebook)

1. **Separação de Dados:** 80% Treino (7.594) / 20% Teste (1.899) estratificado.
2. **Feature Engineering:**
   - Variáveis Numéricas (`tempo_de_casa`): `SimpleImputer(median)` + `StandardScaler`
   - Variáveis Categóricas (8 features): `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`
3. **Desbalanceamento:** Uso de `class_weight='balanced'` e cálculos manuais de pesos nas amostras.
4. **Modelos Testados no Cross-Validation (5-Folds):**
   - Regressão Logística (Baseline)
   - Random Forest Classifier
   - Gradient Boosting Classifier (Vencedor)
5. **Otimização (Tuning):** `RandomizedSearchCV` feito sobre o Gradient Boosting (25 iterações).

---

### 🎯 As 9 Features do Modelo

| # | Feature | Tipo |
|---|---------|------|
| 1 | Área de atuação | Categórica (OneHot) |
| 2 | Função/Cargo | Categórica (OneHot) |
| 3 | Tempo de casa | Numérica (Scaled) |
| 4 | Já utilizou IA | Binária (OneHot) |
| 5 | Atividade principal | Categórica (OneHot) |
| 6 | Objetivo 6 meses | Categórica (OneHot) |
| 7 | Impacto do erro | Categórica (OneHot) |
| 8 | Forma de uso de IA | Categórica (OneHot) |
| 9 | Nível de programação | Categórica (OneHot) |

    """)
