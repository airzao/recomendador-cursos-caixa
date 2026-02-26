import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Plataforma de Direcionamento Estratégico de Capacitação", 
    page_icon="🏦", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Paleta de Cores Institucional
CA_AZUL    = "#0070B8"
CA_ESCURO  = "#003F8A"
CA_LARANJA = "#F5A623"
CA_VERDE   = "#00A859"
CA_CINZA   = "#333333"
CA_CLARO   = "#F0F7FF"
CA_BRANCO  = "#FFFFFF"

# ═══════════════════════════════════════════
# CSS PROFISSIONAL & BLINDAGEM DARK MODE
# ═══════════════════════════════════════════
st.markdown("""
<style>
/* Reset Global */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="block-container"] {
    background-color: #F0F7FF !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Tipografia */
h1, h2, h3 { color: #003F8A !important; }
p, label, span, div { color: #333333 !important; }

/* Banners Específicos (Texto Branco) */
.banner-white, .banner-white * { color: #ffffff !important; }

/* Inputs e Widgets */
[data-testid="stWidgetLabel"] p {
    color: #003F8A !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

/* Selectbox e Dropdowns */
[data-testid="stSelectbox"] > div > div {
    background-color: #ffffff !important;
    border: 1px solid #0070B8 !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
[data-baseweb="select"] * { color: #333333 !important; }
div[data-baseweb="popover"] ul { background-color: #ffffff !important; }
div[data-baseweb="popover"] ul li { color: #333333 !important; }
div[data-baseweb="popover"] ul li:hover { background-color: #E8F4FF !important; color: #0070B8 !important; }

/* Sliders */
div[data-baseweb="slider"] { margin-top: 15px; }

/* Tabs (Abas) */
.stTabs [data-baseweb="tab-list"] { 
    border-bottom: 2px solid #0070B8 !important; 
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] p { 
    color: #666666 !important; 
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] { 
    background-color: #ffffff !important; 
    border-radius: 8px 8px 0 0;
    border: 1px solid #0070B8 !important;
    border-bottom: none !important;
}
.stTabs [aria-selected="true"] p { color: #0070B8 !important; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e0e0e0; }

/* Botão de Ação (CTA) */
[data-testid="baseButton-primary"],
[data-testid="baseButton-primary"] *,
[data-testid="baseButton-primary"] span {
    background: linear-gradient(90deg, #0070B8 0%, #003F8A 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.5rem 1rem;
    box-shadow: 0 4px 12px rgba(0, 63, 138, 0.3);
    transition: all 0.3s ease;
}

[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 63, 138, 0.4);
}

/* Métricas */
[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    padding: 20px !important;
    border: 1px solid #e0e0e0 !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# CARREGAMENTO DO MODELO
# ═══════════════════════════════════════════
@st.cache_resource
def load_models():
    return joblib.load("modelos_comparacao.pkl")

modelos_dict = load_models()

# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════
st.sidebar.markdown(f"<h3 style='color:{CA_ESCURO} !important;'>⚙️ Painel de Controle</h3>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p style='color:{CA_CINZA}; font-size:0.9rem;'>Selecione o motor de IA para simular diferentes comportamentos.</p>", unsafe_allow_html=True)

modelo_selecionado = st.sidebar.radio(
    "Algoritmo Ativo:",
    ["Gradient Boosting", "Random Forest", "Regressão Logística"]
)

# Lógica de Seleção do Modelo
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
    "Gradient Boosting": {"acc": "93,4%", "f1": "92,9%", "top3": "97,7%", "desc": "Alta performance. Detecta padrões complexos não-lineares."},
    "Random Forest":     {"acc": "91,7%", "f1": "91,3%", "top3": "96,5%", "desc": "Modelo robusto, baseado em múltiplas árvores de decisão."},
    "Regressão Logística":{"acc": "88,5%", "f1": "87,4%", "top3": "94,2%", "desc": "Modelo linear base (baseline), com menor capacidade de generalização."}
}

# ═══════════════════════════════════════════
# MAPEAMENTOS (Interface -> Modelo)
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
# HEADER
# ═══════════════════════════════════════════
st.markdown(f"""
<div class="banner-white" 
     style="background:linear-gradient(135deg,{cor_tema} 0%,{CA_ESCURO} 100%);
            padding:32px 40px;border-radius:12px;margin-bottom:24px;
            box-shadow:0 8px 32px rgba(0,63,138,0.2);">
  <h1 style="margin:0;font-size:2.2rem;font-weight:800;letter-spacing:-0.5px;">
    🏦 Plataforma de Direcionamento Estratégico de Capacitação
  </h1>
  <div style="display:flex;align-items:center;gap:12px;margin-top:12px;">
     <span style="background:rgba(255,255,255,0.2);padding:4px 12px;border-radius:20px;font-size:0.85rem;font-weight:600;">Caixa Econômica Federal</span>
     <span style="background:{CA_LARANJA};padding:4px 12px;border-radius:20px;font-size:0.85rem;font-weight:700;color:white;">Modelo: {nome_tecnico}</span>
  </div>
</div>
""", unsafe_allow_html=True)

aba1, aba2, aba3, aba4 = st.tabs(["📋 Perfil & Assessment", "🎯 Recomendação", "📊 Métricas do Modelo", "🎓 Detalhes do Projeto"])

# ══════════════════════════════════════════════════════════════════════
# ABA 1 — PERFIL
# ══════════════════════════════════════════════════════════════════════
with aba1:
    st.markdown(f"<h3 style='color:{CA_ESCURO}; border-bottom:2px solid {CA_AZUL}; padding-bottom:8px; margin-bottom:20px;'>1. Dados Funcionais</h3>", unsafe_allow_html=True)
    
    # CORREÇÃO DO ERRO AQUI: Garantindo 3 colunas para 3 variáveis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.selectbox("🏢 Área de atuação", ["Agencia Varejo","Atendimento","Controladoria","Credito","Financeiro","Juridico","Operacoes","Prevencao a Fraudes","Riscos","TI","Dados e Analytics","PMO","RH","Produtos","Auditoria Interna","Compliance","Seguranca da Informacao"])
    with col2:
        funcao = st.selectbox("👔 Função / Cargo", ["Analista","Coordenador","Desenvolvedor","Especialista","Gestor"])
    with col3:
        tempo_de_casa = st.slider("⏱️ Tempo de casa (anos)", 0, 40, 5)

    st.markdown(f"<h3 style='color:{CA_ESCURO}; border-bottom:2px solid {CA_AZUL}; padding-bottom:8px; margin-top:30px; margin-bottom:20px;'>2. Maturidade Digital</h3>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        ja_utilizou_ia = st.radio("🤖 Já utilizou alguma IA profissionalmente?", ["Sim","Nao"], horizontal=True)
    with col_b:
        nivel_display = st.selectbox("💻 Nível atual de programação", list(map_nivel.keys()))

    st.markdown(f"<h3 style='color:{CA_ESCURO}; border-bottom:2px solid {CA_AZUL}; padding-bottom:8px; margin-top:30px; margin-bottom:20px;'>3. Contexto de Negócio</h3>", unsafe_allow_html=True)

    atividade_display = st.selectbox("📌 Qual tipo de atividade ocupa a maior parte do seu tempo?", list(map_atividade.keys()))
    objetivo_display = st.selectbox("🚀 Qual resultado você mais quer alcançar com IA (6 meses)?", list(map_objetivo.keys()))
    
    c_risk, c_use = st.columns(2)
    with c_risk:
        impacto_display = st.selectbox("⚠️ Se a IA errar no seu contexto, qual o risco?", list(map_impacto.keys()))
    with c_use:
        forma_display = st.selectbox("🎯 Como você pretende usar a IA?", list(map_forma.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Botão centralizado e grande
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("OBTER DIRECIONAMENTO PERSONALIZADO🚀", type="primary", use_container_width=True):
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
            st.success("✅ Perfil processado com sucesso! Acesse a aba 'Recomendação'.")

# ══════════════════════════════════════════════════════════════════════
# ABA 2 — RECOMENDAÇÃO
# ══════════════════════════════════════════════════════════════════════
with aba2:
    if "inp" not in st.session_state:
        st.info("👈 Por favor, preencha o **Perfil** na aba anterior para gerar as recomendações.")
    else:
        st.markdown(f"<h3 style='color:{CA_ESCURO}; margin-bottom:20px;'>🎯 Trilhas Sugeridas pelo Modelo</h3>", unsafe_allow_html=True)
        
        proba = model_ativo.predict_proba(st.session_state["inp"])[0]
        top_idx = np.argsort(-proba)[:3]
        
        medalhas = ["🥇 1ª Opção", "🥈 2ª Opção", "🥉 3ª Opção"]
        fundos = [
            f"linear-gradient(135deg, {cor_tema} 0%, {CA_ESCURO} 100%)",
            f"linear-gradient(135deg, {CA_LARANJA} 0%, #E08B00 100%)",
            f"linear-gradient(135deg, #4A7C59 0%, #2D5A3D 100%)"
        ]
        
        for i, idx in enumerate(top_idx):
            curso = classes[idx]
            conf = proba[idx] * 100
            
            st.markdown(f"""
            <div class="banner-white" style="
                background: {fundos[i]};
                padding: 24px 30px;
                border-radius: 12px;
                margin-bottom: 16px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-left: 6px solid rgba(255,255,255,0.4);
                transition: transform 0.2s;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <p style="font-size:0.9rem; font-weight:600; opacity:0.9; margin:0;">{medalhas[i]}</p>
                        <h2 style="margin:4px 0 0 0; font-size:1.6rem; font-weight:800;">{curso}</h2>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-size:2rem; font-weight:900;">{conf:.0f}%</span>
                        <p style="margin:0; font-size:0.8rem; opacity:0.8;">de aderência</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        
        # Gráfico de probabilidades
        col_chart, col_details = st.columns([2, 1])
        with col_chart:
            st.markdown(f"<h4 style='color:{CA_CINZA};'>📊 Distribuição de Probabilidades</h4>", unsafe_allow_html=True)
            sorted_idx = np.argsort(-proba)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#F0F7FF")
            ax.set_facecolor("#ffffff")
            
            y_pos = np.arange(len(classes))
            scores = [proba[i]*100 for i in sorted_idx]
            names = [classes[i] for i in sorted_idx]
            
            # Cores baseadas no ranking
            bar_colors = [cor_tema if i < 3 else "#CCCCCC" for i in range(len(classes))]
            
            ax.barh(y_pos, scores, align='center', color=bar_colors, height=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=10, color="#333333")
            ax.invert_yaxis()  
            ax.set_xlabel('Probabilidade (%)', fontsize=9, color="#555555")
            
            # Remove bordas desnecessárias
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#DDDDDD')
            
            st.pyplot(fig)

        with col_details:
            st.markdown(f"""
            <div style="background:white; padding:20px; border-radius:10px; border:1px solid #ddd;">
                <h5 style="color:{CA_AZUL}; margin-top:0;">👤 Resumo do Perfil</h5>
                <hr style="margin:10px 0;">
                <p style="font-size:0.9rem;"><b>Área:</b> {st.session_state['inp'].iloc[0]['area']}</p>
                <p style="font-size:0.9rem;"><b>Cargo:</b> {st.session_state['inp'].iloc[0]['funcao']}</p>
                <p style="font-size:0.9rem;"><b>Prog.:</b> {nivel_display}</p>
                <p style="font-size:0.9rem;"><b>Uso IA:</b> {ja_utilizou_ia}</p>
                <p style="font-size:0.8rem; color:#888; margin-top:15px;">*Estas variáveis foram as que mais influenciaram a decisão do modelo.</p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ABA 3 — MÉTRICAS
# ══════════════════════════════════════════════════════════════════════
with aba3:
    dados_kpi = kpis_dinamicos[modelo_selecionado]
    
    st.markdown(f"<h3 style='color:{cor_tema};'>📈 Performance do Modelo ({nome_tecnico})</h3>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:#fff; padding:15px; border-left:4px solid {cor_tema}; border-radius:4px; margin-bottom:20px;'><b>Insight Técnico:</b> {dados_kpi['desc']}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia Global", dados_kpi["acc"])
    c2.metric("Macro F1-Score", dados_kpi["f1"])
    c3.metric("Top-3 Accuracy", dados_kpi["top3"], help="Chance do curso ideal estar entre as 3 primeiras sugestões")
    c4.metric("Dataset", "9.493 linhas")

    st.markdown("---")
    
    st.markdown(f"<h4 style='color:{CA_ESCURO}'>🔬 Detalhes do Treinamento</h4>", unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        **Pipeline de Processamento:**
        1. **Imputação:** Mediana (numéricos) e Moda (categóricos).
        2. **Encoding:** One-Hot Encoding para variáveis categóricas.
        3. **Scaling:** StandardScaler para tempo de casa.
        4. **Balanceamento:** Pesos de classe ajustados (`class_weight='balanced'`).
        """)
    with col_info2:
        st.markdown("""
        **Validação:**
        * **Split:** 80% Treino / 20% Teste (Estratificado)
        * **Cross-Validation:** 5-Folds
        * **Métrica Alvo:** F1-Score Macro (para garantir performance em classes minoritárias).
        """)

# ══════════════════════════════════════════════════════════════════════
# ABA 4 — DETALHES DO PROJETO (COM TABELA HTML E CONTEÚDO RESTAURADO)
# ══════════════════════════════════════════════════════════════════════
with aba4:
    st.markdown(f"<h2 style='color:{CA_AZUL};'>📄 Detalhes do Projeto</h2>", unsafe_allow_html=True)

    # Banner de Resumo (Dinâmico)
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

    # Tabela HTML Personalizada (Aspecto | Detalhe)
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
          <td>Atualmente, na CAIXA, diferentes perfis (ex.: operações, atendimento, riscos, compliance, TI, dados) possuem demandas e usos distintos de IA como automação, agentes, RAG, machine learning, explicabilidade, etc. No entanto, como o tema é novo, os usuários precisam de auxílio para encontrar o treinamento que mais se aproxima de suas necessidades reais. Isso leva a:<br><br>• Baixa aplicação prática após o curso;<br>• Desperdício de investimento em treinamento;<br>• Risco de uso inadequado de IA em contextos críticos (ex.: risco/compliance/jurídico).</td>
        </tr>
        <tr>
          <td><b>Objetivo</b></td>
          <td>Acelerar a adoção de inteligência artificial de forma segura e alinhada às necessidades reais das áreas.</td>
        </tr>
        <tr>
          <td><b>Solução</b></td>
          <td>Será desenvolvido um modelo de <i>machine learning</i> para recomendação de curso/trilha de IA com base em um <i>assessment</i> que considera informações reais dos funcionários (perfil funcional, tipo de atividade, objetivo com IA, impacto do erro, forma de uso e nível de programação).</td>
        </tr>
        <tr>
          <td><b>ROI</b></td>
          <td>O ROI esperado do projeto está na redução de custos e do tempo despendido com treinamentos em IA pouco aderentes, aliada ao aumento da efetividade do uso de IA na Caixa.</td>
        </tr>
        <tr>
          <td><b>Stakeholders</b></td>
          <td>Áreas de negócio usuárias de IA (operações, atendimento, riscos, compliance, TI e dados), além das áreas de RH/L&D, governança de IA e liderança, responsáveis pela capacitação.</td>
        </tr>
        <tr>
          <td><b>Critério de sucesso</b></td>
          <td>O projeto será considerado bem-sucedido quando o modelo recomendar cursos ou trilhas de IA com, no mínimo, 80% de aderência percebida pelos usuários no pós-treinamento, e com taxa de conclusão de no mínimo 70%.</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown(f"<h3 style='color:{CA_ESCURO};'>📂 Dataset de treinamento</h3>", unsafe_allow_html=True)
    st.markdown("""
O dataset de treinamento foi construído a partir de benchmarks derivados de um *assessment* previamente realizado com o objetivo de identificar as principais dores e desafios enfrentados pelos colaboradores. Os dados coletados nesse diagnóstico foram tratados e analisados para mapear lacunas de competências e necessidades de treinamento. A partir desse conjunto inicial, aplicou-se a técnica de geração de dados sintéticos (*data augmentation*) baseada em *Large Language Models* (LLMs), permitindo a criação de novas linhas sintéticas coerentes com os padrões observados, ampliando a representatividade do dataset e fortalecendo a robustez do processo de treinamento.
    """)

    st.markdown(f"<h3 style='color:{CA_ESCURO};'>🛡️ Qualidade do Dataset</h3>", unsafe_allow_html=True)
    st.markdown("""
Foi desenvolvido um notebook de validação e tratamento de qualidade do dataset com o objetivo de assegurar a integridade dos dados utilizados no treinamento. Esse notebook busca garantir que, mesmo após a geração de dados sintéticos, todas as linhas permaneçam aderentes às regras e restrições do modelo (ex.: formatos, domínios permitidos e coerência entre campos), além de identificar e remover registros duplicados e potenciais inconsistências que possam comprometer a performance e a confiabilidade do modelo.
    """)

    st.markdown(f"<h3 style='color:{CA_ESCURO};'>⚙️ Metodologia de projeto (CRISP-DM)</h3>", unsafe_allow_html=True)
    st.markdown("""
O projeto foi desenvolvido seguindo a metodologia **CRISP-DM**, com fases bem definidas e encadeadas, estruturada nas seguintes etapas:

1. **Entendimento do Negócio (Business Understanding)**  
Definição dos objetivos do projeto, escopo, premissas, restrições, critérios de sucesso e entendimento do contexto organizacional e do problema a ser resolvido.
2. **Entendimento dos Dados (Data Understanding)**  
Coleta inicial dos dados, análise exploratória preliminar, identificação da qualidade dos dados, padrões, inconsistências e principais achados do assessment.
3. **Preparação dos Dados (Data Preparation)**  
Limpeza, tratamento, transformação e estruturação dos dados, incluindo:
   * Tratamento de valores ausentes e outliers (`SimpleImputer`).
   * Padronização e enriquecimento das variáveis.
   * Geração de dados sintéticos, quando aplicável.
   * Balanceamento das classes (`class_weight='balanced'`).
4. **Modelagem (Modeling)**  
Seleção de técnicas e algoritmos adequados (Regressão Logística, Random Forest e Gradient Boosting), construção dos modelos analíticos e ajuste de parâmetros (`RandomizedSearchCV`), com base nos objetivos definidos na etapa de negócio.
5. **Avaliação (Evaluation)**  
Validação dos modelos desenvolvidos (Cross-Validation 5-Folds), análise de desempenho frente aos critérios de sucesso estabelecidos (F1-Score Macro e Top-3 Accuracy) e verificação da aderência às necessidades do negócio.
6. **Implantação / Entrega (Deployment)**  
Consolidação dos resultados, organização dos artefatos finais e disponibilização das entregas do projeto (Aplicativo Streamlit em Nuvem).

Como resultado da primeira etapa (Entendimento do Negócio e dos Dados), foi elaborado um documento formal em formato PDF, consolidando os objetivos, escopo, premissas, critérios de sucesso e principais achados do assessment. As etapas técnicas foram documentadas por meio de notebooks em Python (exploração dos dados, tratamento, geração de dados sintéticos, modelagem e validação) garantindo transparência, reprodutibilidade e controle técnico do desenvolvimento do projeto.
    """)

    st.markdown("---")

    st.markdown(f"<h3 style='color:{CA_ESCURO};'>🧬 As 9 Features do Modelo</h3>", unsafe_allow_html=True)
    st.markdown("""
| # | Feature | Tipo | Processamento no Pipeline |
|---|---------|------|---------------------------|
| 1 | Área de atuação | Categórica | `OneHotEncoder` |
| 2 | Função/Cargo | Categórica | `OneHotEncoder` |
| 3 | Tempo de casa | Numérica | `StandardScaler` |
| 4 | Já utilizou IA | Binária | `OneHotEncoder` |
| 5 | Atividade principal | Categórica | `OneHotEncoder` |
| 6 | Objetivo 6 meses | Categórica | `OneHotEncoder` |
| 7 | Impacto do erro | Categórica | `OneHotEncoder` |
| 8 | Forma de uso de IA | Categórica | `OneHotEncoder` |
| 9 | Nível de programação | Categórica | `OneHotEncoder` |
    """)
