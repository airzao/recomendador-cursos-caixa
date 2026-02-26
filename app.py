import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="Recomendador de Cursos — CAIXA", page_icon="🏦", layout="wide", initial_sidebar_state="collapsed")

CG = "#00953B"; CD = "#003D1A"; CL = "#E8F5E9"; CGR = "#4A4A4A"; CW = "#FFFFFF"; CA = "#FFB81C"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background: linear-gradient(135deg, #f5f5f5 0%, #fafafa 100%); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 24px; border-bottom: 3px solid {CG}; }}
.stTabs [aria-selected="true"] {{ color: {CG} !important; border-bottom: 3px solid {CG} !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="background: linear-gradient(135deg, {CG} 0%, {CD} 100%); padding: 40px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 4px 16px rgba(0,149,59,0.3);">
    <h1 style="color:{CW}; margin:0; font-size:2.8rem; font-weight:900;">🏦 Recomendador de Trilhas de IA</h1>
    <p style="color:{CL}; margin:10px 0 0 0; font-size:1.1rem;">Caixa Econômica Federal — Sistema Inteligente de Recomendação</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("modelo_recomendacao_curso.pkl")

model = load_model()
classes = model.named_steps["model"].classes_

aba1, aba2, aba3, aba4 = st.tabs(["📋 Perfil", "🎯 Recomendação Premium", "📊 Modelo & Métricas", "🎓 Apresentação"])

# ── ABA 1 ──────────────────────────────────────────────────────────────────────
with aba1:
    st.markdown(f"<h2 style='color:{CG};'>📋 Perfil do Empregado</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        area = st.selectbox("🏢 Área", ["Agencia Varejo","Atendimento","Controladoria","Credito","Financeiro","Juridico","Operacoes","Prevencao a Fraudes","Riscos","TI"])
        funcao = st.selectbox("👔 Cargo", ["Analista","Coordenador","Desenvolvedor","Especialista","Gestor"])
        tempo_de_casa = st.slider("⏱️ Tempo de casa (anos)", 0, 40, 5)
    with col2:
        ja_utilizou_ia = st.radio("🤖 Já usou IA?", ["Sim","Nao"], horizontal=True)
        nivel_programacao = st.selectbox("💻 Nível programação", ["Nenhum","Basico leio ajusto scripts simples","Intermediario desenvolvo scripts aplicacoes com autonomia","Avancado integracoes debugging boas praticas"])
    col3, col4 = st.columns(2)
    with col3:
        forma_uso_ia = st.selectbox("🎯 Uso de IA", ["Ainda nao sei","Como usuario a de negocio sem programar","Como usuario a com ferramentas no code low code","Como desenvolvedor a programando integracoes solucoes","Como gestor a lider definindo prioridades e direcionando uso"])
        impacto_erro_ia = st.selectbox("⚠️ Impacto do erro", ["Ainda nao sei","Baixo retrabalho pequeno ajuste simples","Medio atraso retrabalho relevante comunicacao incorreta","Alto risco financeiro juridico compliance reputacao decisao"])
    with col4:
        atividade_principal = st.selectbox("📌 Atividade principal", ["Analise de Governaca riscos compliance e controles","Analise para apoio a decisao indicadores desempenho","Atendimento a demandas clientes fornecedores areas internas","Gestao e priorizacao planejamento coordenacao de equipe","Operacao e rotinas administrativas cadastro conferencia","Producao e consolidacao de relatorios apresentacoes","Desenvolvimento de sistemas integracoes APIs"])
        objetivo_ia_6m = st.selectbox("🚀 Objetivo 6 meses", ["Ainda estou explorando quero entender possibilidades","Apoiar decisoes com dashboards BI IA","Automatizar tarefas e fluxos","Classificar organizar informacoes ex documentos tickets","Criar assistentes agentes de IA para apoiar equipes","Entender como a IA funciona","Prever resultados ex demanda risco churn fraude desempenho","Outro"])

    if st.button("💾 GERAR RECOMENDAÇÃO", type="primary", use_container_width=True):
        st.session_state["inp"] = pd.DataFrame([{"area":area,"funcao":funcao,"tempo_de_casa":tempo_de_casa,"ja_utilizou_ia":ja_utilizou_ia,"atividade_principal":atividade_principal,"objetivo_ia_6m":objetivo_ia_6m,"impacto_erro_ia":impacto_erro_ia,"forma_uso_ia":forma_uso_ia,"nivel_programacao":nivel_programacao}])
        st.success("✅ Perfil salvo! Vá para **🎯 Recomendação Premium**.")

# ── ABA 2 ──────────────────────────────────────────────────────────────────────
with aba2:
    st.markdown(f"<h2 style='color:{CG};'>🎯 Trilhas Recomendadas</h2>", unsafe_allow_html=True)
    if "inp" not in st.session_state:
        st.info("👈 Preencha o perfil primeiro.")
    else:
        proba = model.predict_proba(st.session_state["inp"])[0]
        top_idx = np.argsort(-proba)[:3]
        medalhas = ["🥇","🥈","🥉"]
        cores = [f"linear-gradient(135deg, {CG} 0%, {CD} 100%)", f"linear-gradient(135deg, {CA} 0%, #FFA500 100%)", "linear-gradient(135deg, #CD7F32 0%, #8B4513 100%)"]
        borda = ["gold","silver","#CD7F32"]

        for i, idx in enumerate(top_idx):
            curso, conf = classes[idx], proba[idx]*100
            st.markdown(f"""
<div style="background:{cores[i]};color:white;padding:24px;border-radius:12px;margin-bottom:16px;box-shadow:0 6px 20px rgba(0,0,0,0.15);border-left:6px solid {borda[i]};">
  <h3 style="margin:0 0 12px 0;font-size:1.4rem;">{medalhas[i]} {curso}</h3>
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div style="flex:1;background:rgba(255,255,255,0.2);border-radius:8px;height:8px;margin-right:12px;">
      <div style="background:white;height:100%;width:{conf:.0f}%;border-radius:8px;"></div>
    </div>
    <span style="font-size:1.2rem;font-weight:bold;">{conf:.1f}%</span>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown(f"<h3 style='color:{CD};margin-top:20px;'>📊 Confiança para Todos os Cursos</h3>", unsafe_allow_html=True)
        col_v1, col_v2 = st.columns([2,1])
        with col_v1:
            sorted_idx = np.argsort(-proba)
            fig, ax = plt.subplots(figsize=(10,5))
            fig.patch.set_facecolor("#f5f5f5"); ax.set_facecolor("#ffffff")
            ax.barh([classes[i] for i in sorted_idx], [proba[i]*100 for i in sorted_idx],
                    color=[CG if j<3 else CGR for j in range(len(sorted_idx))])
            ax.set_xlabel("Confiança (%)", color=CGR, fontsize=11, fontweight="bold")
            ax.tick_params(colors=CGR, labelsize=10); ax.set_xlim(0,100)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            for i_, idx_ in enumerate(sorted_idx):
                ax.text(proba[idx_]*100+1.5, i_, f"{proba[idx_]*100:.1f}%", va="center", color=CGR, fontsize=9, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        with col_v2:
            st.markdown(f"""
<div style="background:{CL};padding:16px;border-radius:8px;border-left:4px solid {CG};">
  <h4 style="color:{CD};margin-top:0;">💡 Como Ler</h4>
  <p style="margin:0;color:{CGR};font-size:0.9rem;">
    <b>Verde:</b> Top-3 mais indicados<br><br>
    <b>Cinza:</b> Demais cursos<br><br>
    Confiança calculada com base nas <b>9 características</b> do perfil.
  </p>
</div>""", unsafe_allow_html=True)

# ── ABA 3 ──────────────────────────────────────────────────────────────────────
with aba3:
    st.markdown(f"<h2 style='color:{CG};'>📊 Modelo & Métricas</h2>", unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    cards = [
        (m1, "92,8%", "Accuracy", f"linear-gradient(135deg, {CG} 0%, {CD} 100%)"),
        (m2, "92,4%", "Macro F1-Score", f"linear-gradient(135deg, {CA} 0%, #FFA500 100%)"),
        (m3, "99,0%", "Top-3 Accuracy", "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)"),
        (m4, "9.493", "Empregados", "linear-gradient(135deg, #558B2F 0%, #33691E 100%)"),
    ]
    for col_, val_, label_, grad_ in cards:
        with col_:
            st.markdown(f"""
<div style="background:{grad_};padding:20px;border-radius:12px;text-align:center;color:white;margin-bottom:8px;">
  <h1 style="margin:0;font-size:2rem;">{val_}</h1>
  <p style="margin:8px 0 0 0;font-size:0.9rem;">{label_}</p>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<h3 style='color:{CD};'>🧠 Arquitetura — Stacking Classifier</h3>", unsafe_allow_html=True)
    arch1, arch2 = st.columns([1.5,1])
    with arch1:
        st.markdown("""
| Camada | Algoritmo | Papel |
|--------|-----------|-------|
| **Base 1** | Gradient Boosting | Padrões complexos e não-lineares |
| **Base 2** | Random Forest | Robustez e generalização |
| **Base 3** | Logistic Regression | Fronteiras lineares e interpretabilidade |
| **Meta** | Logistic Regression | Combina as 3 predições |
        """)
        st.markdown(f"""
<div style="background:{CL};padding:12px;border-radius:8px;border-left:4px solid {CG};">
<b>Por que Stacking?</b> Cada algoritmo captura um aspecto diferente dos dados. O meta-modelo aprende a combinar os três da melhor forma, resultando em maior acurácia que qualquer modelo isolado.
</div>""", unsafe_allow_html=True)
    with arch2:
        fig2, ax2 = plt.subplots(figsize=(5,4))
        fig2.patch.set_facecolor("#f5f5f5"); ax2.set_facecolor("#f5f5f5")
        ax2.set_xlim(0,10); ax2.set_ylim(0,10); ax2.axis("off")
        bxs = [(0.3,7.2,3.2,1.2,CG,"Gradient Boosting"), (0.3,4.7,3.2,1.2,CA,"Random Forest"), (0.3,2.2,3.2,1.2,CGR,"Logistic Regression"), (5.5,4.2,4.0,1.8,CD,"Meta\nLogistic Regression")]
        for x,y,w,h,cor,lbl in bxs:
            ax2.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.15",facecolor=cor,edgecolor="white",linewidth=2))
            ax2.text(x+w/2,y+h/2,lbl,ha="center",va="center",color="white",fontsize=9,fontweight="bold")
        for y_s in [7.8,5.3,2.8]:
            ax2.annotate("",xy=(5.5,5.1),xytext=(3.5,y_s),arrowprops=dict(arrowstyle="->",color=CG,lw=2.5))
        ax2.annotate("",xy=(9.8,5.1),xytext=(9.5,5.1),arrowprops=dict(arrowstyle="->",color=CG,lw=3))
        ax2.text(9.9,5.1,"🎯",fontsize=14,va="center")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.divider()
    st.markdown(f"<h3 style='color:{CD};'>📂 Dataset & Features</h3>", unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("👥 Empregados","9.493"); d2.metric("📊 Features","9"); d3.metric("🎓 Cursos","8"); d4.metric("✅ Completude","100%")

    cursos_l = ["Fundamentos IA","IA Explicável","Automação","IA Negócios","Prompting","Agentes IA","RAG","ML Negócio"]
    qtds_l = [2375,1768,1231,1129,904,871,774,441]
    fig3, ax3 = plt.subplots(figsize=(10,3.5))
    fig3.patch.set_facecolor("#f5f5f5"); ax3.set_facecolor("#ffffff")
    bars3 = ax3.bar(range(len(cursos_l)),qtds_l,color=CG,alpha=0.85,edgecolor=CD,linewidth=1.5)
    ax3.set_ylabel("Quantidade",color=CGR,fontweight="bold")
    ax3.set_xticks(range(len(cursos_l))); ax3.set_xticklabels(cursos_l,rotation=30,ha="right",fontsize=9,color=CGR)
    ax3.tick_params(colors=CGR); ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    for bar,val in zip(bars3,qtds_l):
        ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+50,str(val),ha="center",va="bottom",color=CD,fontsize=8,fontweight="bold")
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    st.divider()
    st.markdown(f"<h3 style='color:{CD};'>⚙️ Decisões Técnicas</h3>", unsafe_allow_html=True)
    t1,t2 = st.columns(2)
    with t1:
        st.markdown("""
- `class_weight='balanced'` — trata desbalanceamento
- `Pipeline` com `OneHotEncoder` + `StandardScaler`
- Sem Data Leakage (transformação só no treino)
        """)
    with t2:
        st.markdown("""
- `StratifiedKFold` (5 folds) — mantém proporção
- Métrica: `Macro F1-Score` (justo para classes pequenas)
- `joblib compress=3` — modelo < 10 MB
        """)

# ── ABA 4 ──────────────────────────────────────────────────────────────────────
with aba4:
    st.markdown(f"<h2 style='color:{CG};'>🎓 Documentação para Apresentação</h2>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="background: linear-gradient(135deg, {CG} 0%, {CD} 100%); padding: 24px; border-radius: 12px; color: white; margin-bottom: 20px;">
    <h3 style="margin:0 0 8px 0;">📚 Resumo Executivo</h3>
    <p style="margin:0; font-size: 1rem;">
        Sistema de recomendação de trilhas de IA para empregados da Caixa Econômica Federal.<br>
        <b>92,8% de acurácia</b> com Stacking Classifier treinado em 9.493 empregados.
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### 🔍 Problema & Solução

| Aspecto | Detalhe |
|---|---|
| **Problema** | Empregados não sabem qual trilha escolher entre 8 opções |
| **Solução** | Stacking Classifier personalizado por 9 features do perfil |
| **Dataset** | 9.493 empregados reais |
| **Resultado** | 99% dos casos: curso ideal está no Top-3 |

---

### 📊 Metodologia

1. **EDA** — Análise de distribuição, correlações, outliers
2. **Pré-processamento** — OneHotEncoder + StandardScaler em Pipeline
3. **Balanceamento** — class_weight='balanced' (sem oversampling)
4. **Modelagem** — Stacking: GB + RF + LR → meta LR
5. **Validação** — StratifiedKFold 5-Fold (Macro F1)

---

### 🎯 9 Features do Modelo

| # | Feature | Tipo |
|---|---------|------|
| 1 | Área de atuação | Categórica (10) |
| 2 | Função/Cargo | Categórica (5) |
| 3 | Tempo de casa | Numérica |
| 4 | Já usou IA | Binária |
| 5 | Atividade principal | Categórica (7) |
| 6 | Objetivo 6 meses | Categórica (8) |
| 7 | Impacto do erro | Categórica (4) |
| 8 | Forma de uso | Categórica (5) |
| 9 | Nível programação | Categórica (4) |

---

### 📈 Resultados

| Métrica | Valor |
|---|---|
| **Accuracy** | 92,8% |
| **Macro F1-Score** | 92,4% |
| **Top-3 Accuracy** | **99,0%** |
| **Weighted F1** | 92,9% |

---

### 🚀 Deployment

- **Frontend:** Streamlit (interface web interativa)
- **Hosting:** Streamlit Cloud (GitHub → deploy automático)
- **Modelo:** Scikit-learn salvo em `.pkl` (joblib)

---

### 💡 Impacto & Próximos Passos

✅ Facilita escolha para 9.493+ empregados  
✅ Pronto para produção — link público no ar  
✅ Interface intuitiva para não-técnicos  

🔜 **Próximos passos:** Feedback loop, retraining periódico, SHAP para explicabilidade
""")
