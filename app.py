import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Recomendador de Cursos — CAIXA", 
    page_icon="🏦", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

CA_AZUL    = "#0070B8"
CA_ESCURO  = "#003F8A"
CA_LARANJA = "#F5A623"
CA_VERDE   = "#00A859"
CA_CINZA   = "#333333"
CA_CLARO   = "#F0F7FF"
CA_BRANCO  = "#FFFFFF"

# ═══════════════════════════════════════════
# CSS GLOBAL BLINDADO (CONTRA DARK MODE)
# ═══════════════════════════════════════════
st.markdown("""
<style>
/* Fundo e textos gerais */
html, body, [data-testid="stAppViewContainer"],[data-testid="stMain"], [data-testid="block-container"], section.main, .main {
    background-color: #F0F7FF !important;
    color: #333333 !important;
}

/* Forçar texto branco apenas DENTRO dos banners coloridos */
.banner-white, .banner-white h1, .banner-white h2, .banner-white h3, .banner-white h4, .banner-white p, .banner-white span, .banner-white b, .banner-white strong {
    color: #ffffff !important;
}

/* Labels dos inputs */
label, [data-testid="stWidgetLabel"] p {
    color: #333333 !important; 
    font-weight: 600 !important; 
    font-size: 0.95rem !important;
}

/* ======== CORREÇÃO DOS SELECTBOX ======== */
[data-testid="stSelectbox"] > div > div { 
    background-color: #ffffff !important; 
    border: 1.5px solid #0070B8 !important; 
    border-radius: 6px !important; 
}[data-baseweb="select"] * { color: #333333 !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] div, div[data-baseweb="popover"] ul { background-color: #ffffff !important; }
div[data-baseweb="popover"] ul li { color: #333333 !important; background-color: #ffffff !important; }
div[data-baseweb="popover"] ul li:hover { background-color: #E8F4FF !important; color: #0070B8 !important; }

/* ======== CORREÇÃO DAS ABAS ======== */
.stTabs[data-baseweb="tab-list"] { border-bottom: 3px solid #0070B8 !important; background-color: #F0F7FF !important; }
.stTabs[data-baseweb="tab"] { background-color: transparent !important; }
.stTabs [data-baseweb="tab"] p { color: #888888 !important; font-weight: 600 !important; font-size: 1.05rem !important; }
.stTabs[aria-selected="true"] { background-color: #ffffff !important; border-bottom: 3px solid #0070B8 !important; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] p { color: #0070B8 !important; }

/* ======== CORREÇÃO DA SIDEBAR E RADIO ======== */
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #dee2e6; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] p,[data-testid="stSidebar"] span, [data-testid="stSidebar"] label { color: #333333 !important; }
[data-testid="stRadio"] label p { color: #333333 !important; font-weight: 500 !important; }

/* ======== MÉTRICAS E BOTÕES ======== */
[data-testid="metric-container"],[data-testid="stMetric"] { background-color: #ffffff !important; border-radius: 10px !important; padding: 16px !important; border: 1px solid #dee2e6 !important; box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important; }[data-testid="stMetricLabel"] *, [data-testid="metric-container"] label { color: #555555 !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] *, [data-testid="stMetricValue"] { color: #003F8A !important; font-weight: 900 !important; }
[data-testid="baseButton-primary"] { background-color: #0070B8 !important; color: white !important; font-weight: 700 !important; border-radius: 8px !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# LOAD DOS 3 MODELOS
# ═══════════════════════════════════════════
@st.cache_resource
def load_models():
    return joblib.load("modelos_comparacao.pkl")

modelos_dict = load_models()

# ═══════════════════════════════════════════
# SIDEBAR - SIMULADOR DINÂMICO
# ═══════════════════════════════════════════
st.sidebar.markdown(f"<h2 style='color:{CA_AZUL} !important; font-weight: 800;'>🛠️ Simulador de Modelos</h2>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='color:{CA_CINZA}; font-size:0.95rem;'>Escolha o algoritmo abaixo para ver como o aplicativo e as métricas reagem em tempo real.</p>", unsafe_allow_html=True)

modelo_selecionado = st.sidebar.radio(
    "Modelo Ativo:",["Gradient Boosting", "Random Forest", "Regressão Logística"]
)

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

kpis_dinamicos = {
    "Gradient Boosting": {"acc": "93,4%", "f1": "92,9%", "top3": "97,7%", "desc": "Melhor performance geral. Captura relações complexas nos dados e corrige vieses sequencialmente."},
    "Random Forest":        {"acc": "91,7%", "f1": "91,3%", "top3": "96,5%", "desc": "Modelo robusto baseado em múltiplas árvores, mas que apresentou leve queda na métrica Macro F1."},
    "Regressão Logística":  {"acc": "88,5%", "f1": "87,4%", "top3": "94,2%", "desc": "Modelo baseline simples. Funciona bem como base, mas sofre para entender perfis mais complexos."}
}

# ═══════════════════════════════════════════
# MAPEAMENTOS (Display Amigável -> Valor Real do Modelo)
# ═══════════════════════════════════════════
map_atividade = {
    "Análise para apoio à decisão (indicadores, desempenho, relatórios analíticos)": "Analise para apoio a decisao indicadores desempenho relatorios analiticos",
    "Produção e consolidação de relatórios/apresentações": "Producao e consolidacao de relatorios apresentacoes",
    "Operação e rotinas administrativas (cadastro, conferência, execução de processos)": "Operacao e rotinas administrativas cadastro conferencia execucao de processos",
    "Atendimento a demandas (clientes, fornecedores, áreas internas)": "Atendimento a demandas clientes fornecedores areas internas",
    "Desenvolvimento técnico (scripts, integrações, sistemas)": "Desenvolvimento tecnico scripts integracoes sistemas",
    "Gestão e priorização (planejamento, coordenação de equipe, definição de prioridades)": "Gestao e priorizacao planejamento coordenacao de equipe definicao de prioridades",
    "Análise de Governança, riscos, compliance e controles": "Analise de Governanca riscos compliance e controles",
    "Outro": "Outro"
}

map_objetivo = {
    "Entender como a IA funciona": "Entender como a IA funciona",
    "Automatizar tarefas e fluxos": "Automatizar tarefas e fluxos",
    "Melhorar controles, auditoria, risco ou compliance com IA": "Melhorar controles auditoria risco ou compliance com IA",
    "Criar assistentes/agentes de IA para apoiar equipes ou processos": "Criar assistentes agentes de IA para apoiar equipes ou processos",
    "Analisar dados e gerar insights com mais velocidade": "Analisar dados e gerar insights com mais velocidade",
    "Prever resultados (ex.: demanda, risco, churn, fraude, desempenho) usando modelos": "Prever resultados ex demanda risco churn fraude desempenho usando modelos",
    "Classificar/organizar informações (ex.: documentos, tickets, e-mails, textos)": "Classificar organizar informacoes ex documentos tickets e mails textos",
    "Gerar conteúdo (textos, resumos, apresentações, documentação)": "Gerar conteudo textos resumos apresentacoes documentacao",
    "Buscar informação em bases internas (ex.: FAQ, documentos, knowledge base / RAG)": "Buscar informacao em bases internas ex FAQ documentos knowledge base RAG",
    "Apoiar decisões com dashboards/BI + IA": "Apoiar decisoes com dashboards BI IA",
    "Ainda estou explorando / quero entender possibilidades": "Ainda estou explorando quero entender possibilidades",
    "Outro": "Outro"
}

map_impacto = {
    "Baixo (retrabalho pequeno / ajuste simples)": "Baixo retrabalho pequeno ajuste simples",
    "Médio (atraso, retrabalho relevante, comunicação incorreta)": "Medio atraso retrabalho relevante comunicacao incorreta",
    "Alto (risco financeiro, jurídico, compliance, reputação ou decisão crítica)": "Alto risco financeiro juridico compliance reputacao ou decisao critica",
    "Ainda não sei": "Ainda nao sei"
}

map_forma = {
    "Como usuário(a) de negócio (sem programar)": "Como usuario a de negocio sem programar",
    "Como usuário(a) com ferramentas no-code/low-code": "Como usuario a com ferramentas no code low code",
    "Como desenvolvedor(a), programando integrações/soluções": "Como desenvolvedor a programando integracoes solucoes",
    "Como gestor(a)/líder, definindo prioridades e direcionando o uso no time": "Como gestor a lider definindo prioridades e direcionando o uso no time",
    "Ainda não sei": "Ainda nao sei"
}

map_nivel = {
    "Nenhum": "Nenhum",
    "Básico (leio/ajusto scripts simples)": "Basico leio ajusto scripts simples",
    "Intermediário (desenvolvo scripts/aplicações com autonomia)": "Intermediario desenvolvo scripts aplicacoes com autonomia",
    "Avançado (integrações, debugging, boas práticas)": "Avancado integracoes debugging boas praticas"
}

# ═══════════════════════════════════════════
# HEADER UI
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

aba1, aba2, aba3, aba4 = st.tabs(["📋 Perfil", "🎯 Recomendação Premium", "📊 Modelo & Métricas", "🎓 Detalhes do projeto"])

# ══════════════════════════════════════════════════════════════════════
# ABA 1 — PERFIL
# ══════════════════════════════════════════════════════════════════════
with aba1:
    st.markdown(f"<h2 style='color:{CA_AZUL};margin-bottom:4px;'>📋 Perfil do Empregado</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{CA_CINZA};margin-top:0;margin-bottom:20px;'>Preencha os dados abaixo para receber recomendações personalizadas.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        area = st.selectbox("🏢 Área de atuação",["Agencia Varejo","Atendimento","Controladoria","Credito","Financeiro","Juridico","Operacoes","Prevencao a Fraudes","Riscos","TI","Dados e Analytics","PMO","RH","Produtos","Auditoria Interna","Compliance","Seguranca da Informacao"])
    with col2:
        funcao = st.selectbox("👔 Função / Cargo",["Analista","Coordenador","Desenvolvedor","Especialista","Gestor"])
    with col3:
        ja_utilizou_ia = st.radio("🤖 Já utilizou alguma IA?",["Sim","Nao"], horizontal=True)

    tempo_de_casa = st.slider("⏱️ Tempo de casa (anos)", 0, 40, 5)
    st.divider()

    atividade_display = st.selectbox("📌 1) Qual tipo de atividade ocupa a maior parte do seu tempo atualmente?", list(map_atividade.keys()))
    objetivo_display = st.selectbox("🚀 2) Qual resultado você mais quer alcançar com IA no seu trabalho nos próximos 6 meses?", list(map_objetivo.keys()))
    impacto_display = st.selectbox("⚠️ 3) Se a IA errar no seu contexto, qual o risco?", list(map_impacto.keys()))
    forma_display = st.selectbox("🎯 4) Como você pretende usar IA no seu trabalho?", list(map_forma.keys()))
    nivel_display = st.selectbox("💻 5) Qual é seu nível atual de programação?", list(map_nivel.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 GERAR RECOMENDAÇÃO PERSONALIZADA", type="primary", use_container_width=True):
        st.session_state["inp"] = pd.DataFrame([{
            "area": area,
            "funcao": funcao,
            "tempo_de_casa": tempo_de_casa,
            "ja_utilizou_ia": ja_utilizou_ia,
            "atividade_principal": map_atividade[atividade_display],
            "objetivo_ia_6m": map_objetivo[objetivo_display],
            "impacto_erro_ia": map_impacto[impacto_display],
            "forma_uso_ia": map_forma[forma_display],
            "nivel_programacao": map_nivel[nivel_display]
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
        medalhas =["🥇","🥈","🥉"]
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
# ABA 4 — DETALHES DO PROJETO
# ══════════════════════════════════════════════════════════════════════
with aba4:
    st.markdown(f"<h2 style='color:{CA_AZUL};'>📄 Detalhes do Projeto</h2>", unsafe_allow_html=True)

    # Tabela estilizada em Markdown
    st.markdown("""
| Aspecto | Detalhe |
| :--- | :--- |
| **Problema** | Atualmente, na CAIXA, diferentes perfis (ex.: operações, atendimento, riscos, compliance, TI, dados) possuem demandas e usos distintos de IA como automação, agentes, RAG, machine learning, explicabilidade, etc. No entanto, como o tema é novo, os usuários precisam de auxílio para encontrar o treinamento que mais se aproxima de suas necessidades reais. Isso leva a:<br><br>• baixa aplicação prática após o curso;<br>• desperdício de investimento em treinamento;<br>• risco de uso inadequado de IA em contextos críticos (ex.: risco/compliance/jurídico). |
| **Objetivo** | Acelerar a adoção de inteligência artificial de forma segura e alinhada às necessidades reais das áreas. |
| **Solução** | Será desenvolvido um modelo de *machine learning* para recomendação de curso/trilha de IA com base em um *assessment* que considera informações reais dos funcionários (perfil funcional, tipo de atividade, objetivo com IA, impacto do erro, forma de uso e nível de programação). |
| **ROI** | O ROI esperado do projeto está na redução de custos e do tempo despendido com treinamentos em IA pouco aderentes, aliada ao aumento da efetividade do uso de IA na Caixa. |
| **Stakeholders** | Áreas de negócio usuárias de IA (operações, atendimento, riscos, compliance, TI e dados), além das áreas de RH/L&D, governança de IA e liderança, responsáveis pela capacitação. |
| **Critério de sucesso** | O projeto será considerado bem-sucedido quando o modelo recomendar cursos ou trilhas de IA com, no mínimo, 80% de aderência percebida pelos usuários no pós-treinamento, e com taxa de conclusão de no mínimo 70%. |
    """)

    st.markdown("---")
    
    st.markdown(f"<h3 style='color:{CA_ESCURO};'>📂 Dataset de treinamento</h3>", unsafe_allow_html=True)
    st.markdown("""
O dataset de treinamento foi construído a partir de benchmarks derivados de um *assessment* previamente realizado com o objetivo de identificar as principais dores e desafios enfrentados pelos colaboradores. Os dados coletados nesse diagnóstico foram tratados e analisados para mapear lacunas de competências e necessidades de treinamento. A partir desse conjunto inicial, aplicou-se a técnica de geração de dados sintéticos (*data augmentation*) baseada em *Large Language Models* (LLMs), permitindo a criação de novas linhas sintéticas coerentes com os padrões observados, ampliando a representatividade do dataset e fortalecendo a robustez do processo de treinamento.
    """)

    st.markdown(f"<h3 style='color:{CA_ESCURO};'>🛡️ Qualidade do Dataset</h3>", unsafe_allow_html=True)
    st.markdown("""
Foi desenvolvido um notebook de validação e tratamento de qualidade do dataset com o objetivo de assegurar a integridade dos dados utilizados no treinamento. Esse notebook busca garantir que, mesmo após a geração de dados sintéticos, todas as linhas permaneçam aderentes às regras e restrições do modelo (ex.: formatos, domínios permitidos e coerência entre campos), além de identificar e remover registros duplicados e potenciais inconsistências que possam comprometer a performance e a confiabilidade do modelo.
    """)

    st.markdown(f"<h3 style='color:{CA_ESCURO};'>⚙️ Metodologia de projeto</h3>", unsafe_allow_html=True)
    st.markdown("""
O projeto foi desenvolvido seguindo a metodologia **CRISP-DM**, com fases bem definidas e encadeadas, estruturada nas seguintes etapas:

1. **Entendimento do Negócio (Business Understanding)**
Definição dos objetivos do projeto, escopo, premissas, restrições, critérios de sucesso e entendimento do contexto organizacional e do problema a ser resolvido.
2. **Entendimento dos Dados (Data Understanding)**
Coleta inicial dos dados, análise exploratória preliminar, identificação da qualidade dos dados, padrões, inconsistências e principais achados do assessment.
3. **Preparação dos Dados (Data Preparation)**
Limpeza, tratamento, transformação e estruturação dos dados, incluindo:
   * Tratamento de valores ausentes e outliers;
   * Padronização e enriquecimento das variáveis;
   * Geração de dados sintéticos, quando aplicável.
4. **Modelagem (Modeling)**
Seleção de técnicas e algoritmos adequados, construção dos modelos analíticos e ajuste de parâmetros, com base nos objetivos definidos na etapa de negócio.
5. **Avaliação (Evaluation)**
Validação dos modelos desenvolvidos, análise de desempenho frente aos critérios de sucesso estabelecidos e verificação da aderência às necessidades do negócio.
6. **Implantação / Entrega (Deployment)**
Consolidação dos resultados, organização dos artefatos finais e disponibilização das entregas do projeto.

Como resultado da primeira etapa (Entendimento do Negócio e dos Dados), foi elaborado um documento formal em formato PDF, consolidando os objetivos, escopo, premissas, critérios de sucesso e principais achados do assessment. As etapas técnicas foram documentadas por meio de notebooks em Python (exploração dos dados, tratamento, geração de dados sintéticos, modelagem e validação) garantindo transparência, reprodutibilidade e controle técnico do desenvolvimento do projeto.
    """)
