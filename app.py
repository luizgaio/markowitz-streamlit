
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(layout="centered")
st.title("Análise de Carteiras com Fronteira Eficiente")

ativos = st.multiselect("Selecione os ativos para análise:",
                        ['AAPL', 'META', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'],
                        default=['AAPL', 'META', 'TSLA'])

data_inicio = st.date_input("Data de início", value=datetime.today() - timedelta(days=3*365))
data_fim = st.date_input("Data de fim", value=datetime.today())

if len(ativos) >= 2:
    dados = yf.download(ativos + ['^BVSP'], start=data_inicio, end=data_fim)['Close'].dropna()
    retornos = np.log(dados[ativos] / dados[ativos].shift(1)).dropna()
    benchmark = np.log(dados['^BVSP'] / dados['^BVSP'].shift(1)).dropna()

    media = retornos.mean() * 252
    cov = retornos.cov() * 252

    n = 10000
    pesos = np.random.dirichlet(np.ones(len(ativos)), n)
    rets = pesos @ media.values
    riscos = np.sqrt(np.einsum('ij,jk,ik->i', pesos, cov.values, pesos))
    sharpe = rets / riscos

    idx_sharpe_max = np.argmax(sharpe)
    melhor_pesos = pesos[idx_sharpe_max]
    ret_port = retornos @ melhor_pesos

    st.subheader("Carteira com maior Sharpe:")
    for i, ativo in enumerate(ativos):
        st.write(f"{ativo}: {melhor_pesos[i]:.2%}")

    retorno_esperado = ret_port.mean() * 252
    volatilidade = ret_port.std() * np.sqrt(252)
    sharpe_ratio = retorno_esperado / volatilidade
    downside = ret_port[ret_port < 0].std() * np.sqrt(252)
    sortino = retorno_esperado / downside
    covar = np.cov(ret_port, benchmark.loc[ret_port.index])
    beta = covar[0,1] / covar[1,1]
    treynor = retorno_esperado / beta
    var_95 = np.percentile(ret_port, 5) * np.sqrt(252)
    cvar_95 = ret_port[ret_port <= np.percentile(ret_port, 5)].mean() * np.sqrt(252)
    acumulado = (1 + ret_port).cumprod()
    drawdown = acumulado / acumulado.cummax() - 1
    max_dd = drawdown.min()

    st.subheader("Indicadores da Carteira Ótima (Sharpe Máximo)")
    st.markdown(f'''
- **Retorno Esperado (anual)**: {retorno_esperado:.2%}  
- **Volatilidade (anual)**: {volatilidade:.2%}  
- **Sharpe**: {sharpe_ratio:.2f}  
- **Sortino**: {sortino:.2f}  
- **Treynor**: {treynor:.2f}  
- **Beta**: {beta:.2f}  
- **VaR (95%)**: {var_95:.2%}  
- **CVaR (95%)**: {cvar_95:.2%}  
- **Máx. Drawdown**: {max_dd:.2%}  
''')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fronteira Eficiente")
        fig_sim, ax_sim = plt.subplots(figsize=(5, 3))
        sc = ax_sim.scatter(riscos, rets, c=sharpe, cmap='viridis', s=5)
        ax_sim.scatter(riscos[idx_sharpe_max], rets[idx_sharpe_max], c='red', marker='*', s=100)
        ax_sim.set_xlabel("Risco (Volatilidade)")
        ax_sim.set_ylabel("Retorno Esperado")
        plt.colorbar(sc, label="Índice de Sharpe")
        st.pyplot(fig_sim, use_container_width=False)
    with col2:
        st.subheader("Desempenho Acumulado (Base 100)")
        ibov = benchmark.loc[ret_port.index]
        base100_port = (1 + ret_port).cumprod() * 100
        base100_ibov = (1 + ibov).cumprod() * 100
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(base100_port, label="Carteira")
        ax1.plot(base100_ibov, label="Ibovespa")
        ax1.legend()
        st.pyplot(fig1, use_container_width=False)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Volatilidade Móvel (30 dias)")
        vol_port = ret_port.rolling(30).std() * np.sqrt(252)
        vol_ibov = ibov.rolling(30).std() * np.sqrt(252)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.plot(vol_port, label="Carteira")
        ax2.plot(vol_ibov, label="Ibovespa")
        ax2.legend()
        st.pyplot(fig2, use_container_width=False)
    with col4:
        st.subheader("Drawdown Comparado")
        dd_ibov = (1 + ibov).cumprod()
        dd_ibov = dd_ibov / dd_ibov.cummax() - 1
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.plot(drawdown, label="Carteira")
        ax3.plot(dd_ibov, label="Ibovespa")
        ax3.legend()
        st.pyplot(fig3, use_container_width=False)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Alocação de Ativos")
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.bar(ativos, melhor_pesos)
        ax4.set_ylabel('Peso na Carteira')
        st.pyplot(fig4, use_container_width=False)
    with col6:
        st.subheader("Matriz de Correlação")
        fig5, ax5 = plt.subplots(figsize=(5, 3))
        sns.heatmap(retornos.corr(), annot=True, cmap='coolwarm', ax=ax5)
        st.pyplot(fig5, use_container_width=False)

st.markdown("---")
st.markdown("Desenvolvido pelo **Prof. Luiz Eduardo Gaio** para fins educacionais.")
