
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Layout e estilo
st.set_page_config(page_title="Markowitz App", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
        .css-18e3th9 {background-color: #101010;}
        .css-1dp5vir {color: #FFFFFF;}
        .reportview-container .markdown-text-container {
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Título
st.markdown("<h1 style='color:#191970;'>Análise de Carteiras com Fronteira Eficiente</h1>", unsafe_allow_html=True)

# Entrada de dados
ativos = st.multiselect("Selecione os ativos:", ['AAPL', 'META', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'], default=['AAPL', 'META', 'TSLA'])

col1, col2 = st.columns(2)
with col1:
    data_inicio = st.date_input("Data de início", value=datetime.today() - timedelta(days=3*365))
with col2:
    data_fim = st.date_input("Data de fim", value=datetime.today())

if len(ativos) >= 2:
    dados = yf.download(ativos + ['^BVSP'], start=data_inicio, end=data_fim)['Close'].dropna()
    retornos = np.log(dados[ativos] / dados[ativos].shift(1)).dropna()
    benchmark = np.log(dados['^BVSP'] / dados['^BVSP'].shift(1)).dropna()

    media = retornos.mean() * 252
    cov = retornos.cov() * 252

    n = 5000
    pesos = np.random.dirichlet(np.ones(len(ativos)), n)
    rets = pesos @ media.values
    riscos = np.sqrt(np.einsum('ij,jk,ik->i', pesos, cov.values, pesos))
    sharpe = rets / riscos

    idx_sharpe_max = np.argmax(sharpe)
    melhor_pesos = pesos[idx_sharpe_max]
    ret_port = (retornos @ melhor_pesos)

    # Métricas
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

    st.markdown("## Carteira Ótima (Sharpe Máximo)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""- **Retorno Esperado (anual)**: {retorno_esperado:.2%}  
- **Volatilidade (anual)**: {volatilidade:.2%}  
- **Sharpe**: {sharpe_ratio:.2f}  
- **Sortino**: {sortino:.2f}  
- **Treynor**: {treynor:.2f}""")
    with col2:
        st.markdown(f"""- **Beta**: {beta:.2f}  
- **VaR (95%)**: {var_95:.2%}  
- **CVaR (95%)**: {cvar_95:.2%}  
- **Máx. Drawdown**: {max_dd:.2%}""")

    df_fronteira = pd.DataFrame({'Retorno': rets, 'Risco': riscos, 'Sharpe': sharpe})
    fig_fronteira = px.scatter(df_fronteira, x='Risco', y='Retorno', color='Sharpe',
                               title="Fronteira Eficiente", color_continuous_scale='Viridis')
    fig_fronteira.add_trace(go.Scatter(x=[riscos[idx_sharpe_max]], y=[rets[idx_sharpe_max]],
                                       mode='markers', marker=dict(color='red', size=12),
                                       name='Máx. Sharpe'))
    st.plotly_chart(fig_fronteira, use_container_width=True)

    ibov = benchmark.loc[ret_port.index]
    base100_port = (1 + ret_port).cumprod() * 100
    base100_ibov = (1 + ibov).cumprod() * 100
    fig_acum = go.Figure()
    fig_acum.add_trace(go.Scatter(y=base100_port, name="Carteira"))
    fig_acum.add_trace(go.Scatter(y=base100_ibov, name="Ibovespa"))
    fig_acum.update_layout(title="Desempenho Acumulado (Base 100)")
    st.plotly_chart(fig_acum, use_container_width=True)

    vol_port = ret_port.rolling(30).std() * np.sqrt(252)
    vol_ibov = ibov.rolling(30).std() * np.sqrt(252)
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(y=vol_port, name="Carteira"))
    fig_vol.add_trace(go.Scatter(y=vol_ibov, name="Ibovespa"))
    fig_vol.update_layout(title="Volatilidade Móvel (30 dias)")
    st.plotly_chart(fig_vol, use_container_width=True)

    dd_ibov = (1 + ibov).cumprod()
    dd_ibov = dd_ibov / dd_ibov.cummax() - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(y=drawdown, name="Carteira"))
    fig_dd.add_trace(go.Scatter(y=dd_ibov, name="Ibovespa"))
    fig_dd.update_layout(title="Drawdown Comparado")
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("### Alocação da Carteira Ótima")
    fig_pesos = px.bar(x=ativos, y=melhor_pesos, labels={'x': 'Ativo', 'y': 'Peso'},
                       title="Distribuição de Pesos na Carteira")
    st.plotly_chart(fig_pesos, use_container_width=True)

    st.markdown("### Matriz de Correlação dos Ativos")
    corr = retornos.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                         title="Matriz de Correlação")
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.markdown("<center>Desenvolvido pelo <strong>Prof. Luiz Eduardo Gaio</strong> para fins educacionais</center>",
            unsafe_allow_html=True)
