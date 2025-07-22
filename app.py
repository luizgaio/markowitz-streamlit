
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

st.set_page_config(layout="wide")
st.title("Análise de Carteiras com Fronteira Eficiente")

# Seleção de ativos
ativos = st.multiselect(
    "Selecione os ativos para análise:",
    ["AAPL", "MSFT", "GOOG", "META", "AMZN", "TSLA", "BRK-B", "JPM", "NVDA", "JNJ"],
    default=["AAPL", "MSFT", "GOOG", "META"]
)

# Coleta de dados
if ativos:
    dados = yf.download(ativos, period="3y")["Adj Close"]
    retornos = np.log(dados / dados.shift(1)).dropna()
    media = retornos.mean() * 252
    cov = retornos.cov() * 252

    # Simulação de carteiras
    n_carteiras = 10000
    pesos = np.random.dirichlet(np.ones(len(ativos)), n_carteiras)
    rets = pesos @ media.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov.values, pesos))
    sharpe = rets / vols

    # Carteiras ótimas
    idx_mv = vols.argmin()
    idx_ms = sharpe.argmax()

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(vols, rets, c=sharpe, cmap='viridis', alpha=0.5)
    ax.scatter(vols[idx_mv], rets[idx_mv], c='green', s=100, label="Mínima Variância")
    ax.scatter(vols[idx_ms], rets[idx_ms], c='red', s=100, label="Máximo Sharpe")
    ax.set_title("Risco vs Retorno com Fronteira Eficiente")
    ax.set_xlabel("Volatilidade Anualizada")
    ax.set_ylabel("Retorno Anualizado")
    ax.legend()
    st.pyplot(fig)

    # Correlação
    st.subheader("Matriz de Correlação dos Retornos")
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(retornos.corr(), annot=True, cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)

    # Pesos da carteira de máximo Sharpe
    st.subheader("Pesos da Carteira de Máximo Sharpe")
    pesos_df = pd.DataFrame(pesos[idx_ms], index=ativos, columns=["Peso"])
    st.bar_chart(pesos_df)

else:
    st.warning("Selecione pelo menos um ativo.")
