import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Título
st.title("Análise de Carteiras com Fronteira Eficiente")

# Seleção dos ativos
ativos = st.multiselect(
    "Selecione os ativos para análise:",
    ['AAPL', 'META', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'],
    default=['AAPL', 'META', 'TSLA']
)

# Coleta de dados
if len(ativos) >= 2:
    dados = yf.download(ativos, start="2022-01-01")['Close']
    retornos = np.log(dados / dados.shift(1)).dropna()

    media = retornos.mean()
    cov = retornos.cov()

    n = 10_000
    n_ativos = len(ativos)
    pesos = np.random.dirichlet(np.ones(n_ativos), n)

    retornos_esperados = pesos @ media.values
    riscos = np.sqrt(np.einsum('ij,jk,ik->i', pesos, cov.values, pesos))
    sharpe = retornos_esperados / riscos

    idx_sharpe_max = np.argmax(sharpe)
    melhor_pesos = pesos[idx_sharpe_max]

    st.subheader("Carteira com maior Sharpe:")
    for i, ativo in enumerate(ativos):
        st.write(f"{ativo}: {melhor_pesos[i]:.2%}")

    fig1, ax1 = plt.subplots()
    sc = ax1.scatter(riscos, retornos_esperados, c=sharpe, cmap='viridis', s=5)
    ax1.scatter(riscos[idx_sharpe_max], retornos_esperados[idx_sharpe_max], c='red', marker='*', s=100)
    ax1.set_xlabel("Risco (Volatilidade)")
    ax1.set_ylabel("Retorno Esperado")
    ax1.set_title("Fronteira Eficiente")
    plt.colorbar(sc, label="Índice de Sharpe")
    st.pyplot(fig1)

    st.subheader("Matriz de Correlação")
    fig2, ax2 = plt.subplots()
    sns.heatmap(retornos.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Drawdown da Carteira Ótima")
    carteira_retorno = (retornos * melhor_pesos).sum(axis=1)
    carteira_cumulativo = (1 + carteira_retorno).cumprod()
    max_cumulativo = carteira_cumulativo.cummax()
    drawdown = carteira_cumulativo / max_cumulativo - 1

    fig3, ax3 = plt.subplots()
    ax3.plot(drawdown, label="Drawdown")
    ax3.set_title("Drawdown da Carteira Ótima")
    st.pyplot(fig3)

else:
    st.warning("Selecione pelo menos dois ativos para análise.")
