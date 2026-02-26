import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Recomendador de Cursos — CAIXA", page_icon="🏦", layout="centered")
st.title("🏦 Recomendador de Trilhas de IA — CAIXA")

@st.cache_resource
def load_model():
    return joblib.load("modelo_recomendacao_curso.pkl")

model   = load_model()
classes = model.named_steps["model"].classes_

aba1, aba2, aba3 = st.tabs(["📋 Perfil", "🎯 Recomendação", "ℹ️ Sobre o Modelo"])

# ─── ABA 1 — PERFIL ──────────────────────────────────────────────────────────
with aba1:
    st.subheader("Preencha o perfil do empregado")

    area = st.selectbox("Área de atuação", [
        "Agencia Varejo", "Atendimento", "Controladoria",
        "Credito", "Financeiro", "Juridico", "Operacoes",
        "Prevencao a Fraudes", "Riscos", "TI"
    ])

    funcao = st.selectbox("Função / Cargo", [
        "Analista", "Coordenador", "Desenvolvedor",
        "Especialista", "Gestor"
    ])

    tempo_de_casa = st.slider("Tempo de casa (anos)", 0, 40, 5)

    ja_utilizou_ia = st.radio("Já utilizou alguma IA?", ["Sim", "Nao"], horizontal=True)

    nivel_programacao = st.selectbox("Nível de programação", [
        "Nenhum",
        "Basico leio ajusto scripts simples",
        "Intermediario desenvolvo scripts aplicacoes com autonomia",
        "Avancado integracoes debugging boas praticas"
    ])

    forma_uso_ia = st.selectbox("Como pretende usar IA?", [
        "Ainda nao sei",
        "Como usuario a de negocio sem programar",
        "Como usuario a com ferramentas no code low code",
        "Como desenvolvedor a programando integracoes solucoes",
        "Como gestor a lider definindo prioridades e direcionando uso"
    ])

    impacto_erro_ia = st.selectbox("Impacto se a IA errar", [
        "Ainda nao sei",
        "Baixo retrabalho pequeno ajuste simples",
        "Medio atraso retrabalho relevante comunicacao incorreta",
        "Alto risco financeiro juridico compliance reputacao decisao"
    ])

    atividade_principal = st.selectbox("Atividade principal no dia a dia", [
        "Analise de Governaca riscos compliance e controles",
        "Analise para apoio a decisao indicadores desempenho",
        "Atendimento a demandas clientes fornecedores areas internas",
        "Gestao e priorizacao planejamento coordenacao de equipe",
        "Operacao e rotinas administrativas cadastro conferencia",
        "Producao e consolidacao de relatorios apresentacoes",
        "Desenvolvimento de sistemas integracoes APIs"
    ])

    objetivo_ia_6m = st.selectbox("O que quer fazer com IA nos próximos 6 meses?", [
        "Ainda estou explorando quero entender possibilidades",
        "Apoiar decisoes com dashboards BI IA",
        "Automatizar tarefas e fluxos",
        "Classificar organizar informacoes ex documentos tickets",
        "Criar assistentes agentes de IA para apoiar equipes",
        "Entender como a IA funciona",
        "Prever resultados ex demanda risco churn fraude desempenho",
        "Outro"
    ])

    if st.button("💾 Salvar perfil e ver recomendação", type="primary"):
        st.session_state["inp"] = pd.DataFrame([{
            "area":               area,
            "funcao":             funcao,
            "tempo_de_casa":      tempo_de_casa,
            "ja_utilizou_ia":     ja_utilizou_ia,
            "atividade_principal": atividade_principal,
            "objetivo_ia_6m":     objetivo_ia_6m,
            "impacto_erro_ia":    impacto_erro_ia,
            "forma_uso_ia":       forma_uso_ia,
            "nivel_programacao":  nivel_programacao
        }])
        st.success("✅ Perfil salvo! Vá para a aba **Recomendação**.")

# ─── ABA 2 — RECOMENDAÇÃO ────────────────────────────────────────────────────
with aba2:
    st.subheader("🎯 Cursos recomendados para este perfil")

    if "inp" not in st.session_state:
        st.info("👈 Preencha o perfil na aba **Perfil** primeiro.")
    else:
        proba   = model.predict_proba(st.session_state["inp"])[0]
        top_idx = np.argsort(-proba)[:3]

        medalhas = ["🥇", "🥈", "🥉"]
        for i, idx in enumerate(top_idx):
            curso     = classes[idx]
            confianca = proba[idx] * 100
            st.markdown(f"### {medalhas[i]} {curso}")
            st.progress(int(confianca), text=f"{confianca:.1f}% de confiança")
            st.divider()

        st.caption("O modelo analisa 9 características do perfil e retorna os 3 cursos mais adequados.")

# ─── ABA 3 — SOBRE O MODELO ──────────────────────────────────────────────────
with aba3:
    st.subheader("ℹ️ Sobre o Modelo")

    st.markdown("""
    **Algoritmo:** Stacking Classifier (ensemble de 3 modelos)

    | Camada | Modelo | Papel |
    |--------|--------|-------|
    | Base 1 | Gradient Boosting | Captura padrões complexos |
    | Base 2 | Random Forest | Robustez e generalização |
    | Base 3 | Logistic Regression | Fronteiras lineares |
    | Meta   | Logistic Regression | Combina as predições |

    **Dataset:** 9.493 empregados · 9 features · 8 cursos

    **Métricas no test set:**
    - Accuracy: ~92%
    - Macro F1-Score: ~92%
    - **Top-3 Accuracy: ~99%**

    > Em 99% dos casos, o curso ideal está entre os 3 recomendados.
    """)
