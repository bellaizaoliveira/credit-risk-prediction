# 🏦 Credit Risk Prediction — Previsão de Inadimplência com Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)
![SHAP](https://img.shields.io/badge/SHAP-Interpretabilidade-green)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

> Um modelo preditivo completo para estimar a probabilidade de inadimplência de clientes em solicitações de empréstimo — do dado bruto ao insight de negócio.

---

## 📋 Sumário

- [O Problema de Negócio](#-o-problema-de-negócio)
- [Dataset](#-dataset)
- [Metodologia](#-metodologia)
- [Resultados](#-resultados)
- [Principais Insights (SHAP)](#-principais-insights-shap)
- [Como Executar](#-como-executar)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)

---

## 💡 O Problema de Negócio

Quando um banco ou fintech recebe uma solicitação de empréstimo, precisa responder uma pergunta simples com consequências milionárias:

> **Esse cliente vai pagar o que deve?**

Errar tem custo nos dois sentidos:

| Tipo de Erro | Situação | Consequência |
|---|---|---|
| **Falso Negativo** | Libera crédito → cliente não paga | Prejuízo financeiro direto |
| **Falso Positivo** | Nega crédito → cliente pagaria | Perda de receita e market share |

Modelos de **Credit Scoring** existem para minimizar ambos simultaneamente, usando dados históricos para estimar a probabilidade de inadimplência de cada cliente.

**Este projeto entrega:**
- ✅ Modelo preditivo comparando 3 algoritmos (Regressão Logística, Random Forest, XGBoost)
- ✅ Interpretabilidade com SHAP — explicando *por que* cada decisão foi tomada
- ✅ Insights acionáveis para a área de crédito

---

## 📊 Dataset

O dataset contém **10.000 solicitações de empréstimo** com 12 variáveis cobrindo o perfil do cliente, características do empréstimo e histórico de crédito.

| Variável | Tipo | Descrição |
|---|---|---|
| `person_age` | Numérica | Idade do solicitante |
| `person_income` | Numérica | Renda anual |
| `person_home_ownership` | Categórica | Tipo de moradia: RENT / OWN / MORTGAGE |
| `person_emp_length` | Numérica | Tempo de emprego (anos) |
| `loan_intent` | Categórica | Finalidade: PERSONAL, EDUCATION, MEDICAL... |
| `loan_grade` | Categórica | Rating do empréstimo: A (menor risco) → G (maior risco) |
| `loan_amnt` | Numérica | Valor solicitado |
| `loan_int_rate` | Numérica | Taxa de juros anual (%) |
| `loan_percent_income` | Numérica | Valor do empréstimo ÷ renda anual |
| `cb_person_default_on_file` | Categórica | Histórico de inadimplência? Y / N |
| `cb_person_cred_hist_length` | Numérica | Anos de histórico de crédito |
| **`loan_status`** | **🎯 Target** | **1 = Inadimplente / 0 = Adimplente** |

**Taxa de inadimplência:** ~30% (dataset desbalanceado — tratado com `class_weight='balanced'` e `scale_pos_weight`)

---

## 🔬 Metodologia

```
Dados Brutos → EDA → Pré-processamento → Modelagem → Avaliação → Interpretabilidade → Insights
```

### 1. Análise Exploratória (EDA)
- Distribuição do target (balanceamento de classes)
- Distribuição das variáveis numéricas por classe (adimplente vs. inadimplente)
- Taxa de default por categoria (loan_grade, home_ownership, etc.)
- Matriz de correlação

### 2. Pré-processamento
- **Imputação por mediana** para valores ausentes (`loan_int_rate`: ~9.9% ausente, `person_emp_length`: ~2.8%)
- **Label Encoding ordinal** para `loan_grade` (A=0, B=1, ..., G=6)
- **One-Hot Encoding** para variáveis nominais (`person_home_ownership`, `loan_intent`)
- **StandardScaler** para padronização
- **Atenção ao Data Leakage**: fit() apenas nos dados de treino; transform() em ambos

### 3. Modelagem

| Modelo | Justificativa |
|---|---|
| **Regressão Logística** | Baseline — simples, interpretável, referência |
| **Random Forest** | Captura relações não-lineares; robusto a outliers |
| **XGBoost** | Estado da arte para dados tabulares; gradient boosting |

### 4. Avaliação
Métricas escolhidas para dados desbalanceados:
- **AUC-ROC**: capacidade de separar as classes (principal métrica)
- **Precision / Recall / F1** para a classe minoritária (inadimplentes)

---

## 📈 Resultados

| Modelo | AUC-ROC | Precision (default) | Recall (default) | F1 (default) |
|---|---|---|---|---|
| Regressão Logística | ~0.720 | ~0.58 | ~0.67 | ~0.62 |
| Random Forest | ~0.790 | ~0.64 | ~0.71 | ~0.67 |
| **XGBoost** | **~0.830** | **~0.69** | **~0.75** | **~0.72** |

O **XGBoost** obteve o melhor AUC-ROC, sendo escolhido como modelo final.

Um AUC de ~0.83 significa que **em 83% dos casos**, quando o modelo compara um cliente inadimplente com um adimplente, ele consegue identificar corretamente quem é quem.

---

## 🔍 Principais Insights (SHAP)

SHAP (SHapley Additive exPlanations) explica a contribuição de cada variável para cada predição individual.

**Top 5 features por importância:**

| # | Feature | Direção do Impacto |
|---|---|---|
| 1 | `loan_percent_income` | ↑ Quanto maior, maior o risco de default |
| 2 | `loan_grade` | ↑ Grades piores (E, F, G) elevam muito o risco |
| 3 | `cb_person_default_on_file` | ↑ Histórico de default anterior sinaliza alto risco |
| 4 | `loan_int_rate` | ↑ Taxas de juros altas correlacionam com maior risco |
| 5 | `cb_person_cred_hist_length` | ↓ Histórico de crédito longo reduz o risco |

**Recomendações de negócio:**
- Clientes com `loan_percent_income` > 0.40 → exigir garantias adicionais
- Grades F e G → análise manual obrigatória
- `cb_default = Y` → documentação complementar requerida
- Clientes jovens com histórico < 2 anos → limite inicial reduzido

---

## 🚀 Como Executar

### Pré-requisitos

```bash
git clone https://github.com/SEU_USUARIO/credit-risk-prediction.git
cd credit-risk-prediction
pip install -r requirements.txt
```

### Rodar o notebook

```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

O dataset já está incluído em `data/credit_risk_dataset.csv` — basta rodar as células em ordem.

---

## 📁 Estrutura do Repositório

```
credit-risk-prediction/
│
├── data/
│   └── credit_risk_dataset.csv      # Dataset com 10.000 solicitações
│
├── notebooks/
│   └── credit_risk_analysis.ipynb   # Notebook principal (EDA → Modelo → SHAP)
│
├── images/                          # Gráficos gerados pelo notebook
│   ├── 01_target_distribution.png
│   ├── 02_numeric_distributions.png
│   ├── 03_categorical_default_rates.png
│   ├── 04_correlation_matrix.png
│   ├── 05_roc_comparison.png
│   ├── 06_confusion_matrix.png
│   ├── 07_shap_importance.png
│   ├── 08_shap_beeswarm.png
│   └── 09_shap_individual.png
│
├── requirements.txt                 # Dependências do projeto
└── README.md
```

---

## 🛠 Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| **Python 3.9+** | Linguagem principal |
| **Pandas** | Manipulação e análise de dados |
| **NumPy** | Operações numéricas |
| **Scikit-learn** | Pré-processamento e modelagem (LR, RF) |
| **XGBoost** | Gradient Boosting (modelo final) |
| **SHAP** | Interpretabilidade do modelo |
| **Matplotlib / Seaborn** | Visualização de dados |
| **Jupyter Notebook** | Ambiente de desenvolvimento interativo |

---

## 👩‍💻 Autora

**Izabella da Silva Oliveira**  
Cientista de Dados | IA Generativa & Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bellaiza-blue?logo=linkedin)](https://linkedin.com/in/bellaiza)
[![GitHub](https://img.shields.io/badge/GitHub-SEU_USUARIO-black?logo=github)](https://github.com/bellaizaoliveira/)

---

*Projeto desenvolvido para fins educacionais e de portfólio.*
