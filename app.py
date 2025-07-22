import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Layout e estilo
st.set_page_config(page_title="Markowitz App", layout="wide", initial_sidebar_state="expanded")

# Título
st.markdown("<h1 style='color:#191970;'>Análise de Portfólio</h1>", unsafe_allow_html=True)


# Entrada de dados
# ativos = st.multiselect("Selecione os ativos:", ['AAPL', 'META', 'TSLA', 'MSFT', 'GOOGL', 'AMZN'], default=['AAPL', 'META', 'TSLA'])
ativos_brasil = [
    "ALPK3", "DOHL3", "GOLL4", "AMBP3", "DOHL4", "VSTE3", "SUZB3", "FRIO3", "OPCT3", "TUPY3", "FIEI3", "PETZ3", "CASH3",
    "DTCY3", "IRBR3", "SCAR3", "HAPV3", "AVLL3", "MAPT4", "RAIL3", "MOAR3", "JALL3", "AURA33", "MAPT3", "CVCB3",
    "SIMH3", "BIOM3", "AALR3", "ENJU3", "ZAMP3", "HBSA3", "ATED3", "DOTZ3", "NGRD3", "PDTC3", "BLUT3", "LJQQ3",
    "RPAD5", "PTBL3", "MRVE3", "RNEW11", "RNEW3", "CRPG6", "RNEW4", "MLAS3", "CRPG3", "ONCO3", "MEAL3", "BEEF3",
    "MATD3", "ADHM3", "AMAR3", "CSNA3", "CRPG5", "RAIZ4", "RPAD3", "RCSL3", "TRAD3", "FICT3", "AMOB3", "GFSA3",
    "RPAD6", "AZTE3", "WEST3", "NTCO3", "RCSL4", "BLUT4", "LUPA3", "DASA3", "BIED3", "GPIV33", "AZEV4", "AZEV3",
    "ESTR4", "CEED3", "CSAN3", "TCSA3", "TXRX3", "MGEL4", "PCAR3", "BRKM3", "TOKY3", "BRKM6", "BRKM5", "VIVR3",
    "RDNI3", "SNSY5", "SHOW3", "BDLL3", "PLAS3", "BDLL4", "INEP3", "INEP4", "AERI3", "VVEO3", "BHIA3", "FHER3",
    "ATMP3", "CTAX3", "TXRX4", "CTSA3", "CTSA4", "RPMG3", "JFEN3", "PMAM3", "IFCM3", "SEQL3", "AGXY3", "GSHP3",
    "PDGR3", "OSXB3", "OIBR3", "AZUL4", "NEXP3", "AMER3", "RSID3", "OIBR4", "IGBR3", "HOOT4", "LIGT3", "MWET4",
    "NORD3", "MWET3", "CEDO4", "CTKA3", "CTKA4", "SYNE3", "CEDO3", "AHEB3", "JBSS3", "JHSF3", "HETA4", "BAZA3",
    "EALT4", "PRIO3", "BBAS3", "EALT3", "VTRU3", "VBBR3", "SOND6", "BNBR3", "ISAE4", "BGIP4", "BMEB3", "CMIG4",
    "AHEB5", "SOND5", "SAPR4", "BRSR6", "SAPR11", "LOGG3", "LUXM4", "BMEB4", "SAPR3", "PINE3", "BMGB4", "TKNO4",
    "BRSR3", "PINE4", "COGN3", "BALM4", "CLSC3", "BRAP3", "SBFG3", "ABCB4", "BGIP3", "BOBR4", "HBOR3", "RANI3",
    "ENGI4", "MRSA5B", "CAMB3", "CLSC4", "CGRA4", "CEEB3", "BRAP4", "CEEB5", "ECOR3", "ISAE3", "ENGI11", "CGRA3",
    "PTNT4", "ALLD3", "CYRE3", "VLID3", "COCE3", "TECN3", "DEXP4", "MRSA6B", "CMIG3", "DEXP3", "MDNE3", "EQPA3",
    "EQMA3B", "GRND3", "MRSA3B", "MTRE3", "COCE5", "TRIS3", "LAVV3", "EZTC3", "EUCA4", "POMO3", "WLMM4", "BRSR5",
    "BALM3", "AGRO3", "SANB3", "LVTC3", "SHUL4", "TAEE4", "BBDC3", "TAEE11", "TAEE3", "WIZC3", "MRFG3", "RSUL4",
    "ENGI3", "BEES4", "WLMM3", "WHRL4", "CSMG3", "BEES3", "SANB11", "RECV3", "NUTR3", "HAGA4", "WHRL3", "JSLG3",
    "RAPT3", "EQPA5", "SANB4", "SBSP3", "EUCA3", "FIQE3", "KEPL3", "PLPL3", "RAPT4", "ITSA4", "ITSA3", "CEBR3",
    "FESA4", "CEBR5", "UGPA3", "LEVE3", "PFRM3", "BBSE3", "CMIN3", "BBDC4", "CSUD3", "VALE3", "EQPA6", "NEOE3",
    "CBEE3", "MOVI3", "TGMA3", "BPAC5", "ETER3", "CPFE3", "POMO4", "GOAU4", "VIVA3", "GOAU3", "CEBR6", "PETR4",
    "ALUP4", "CAML3", "BLAU3", "HBRE3", "TTEN3", "UNIP3", "ALUP11", "PATI4", "ITUB3", "TPIS3", "VULC3", "ROMI3",
    "BRBI11", "PETR3", "MILS3", "EQPA7", "ALUP3", "CSED3", "CRFB3", "ELET3", "DMVF3", "MYPK3", "GGBR3", "EKTR3",
    "BRFS3", "CGAS3", "CGAS5", "UNIP6", "ITUB4", "UNIP5", "BMIN4", "OFSA3", "TASA4", "MULT3", "LPSB3", "ELET6",
    "EGIE3", "VITT3", "ARML3", "GGBR4", "MTSA4", "REDE3", "SMTO3", "DESK3", "CXSE3", "DIRR3", "SLCE3", "PATI3",
    "ODPV3", "INTB3", "HAGA3", "PTNT3", "CPLE3", "YDUQ3", "JOPA3", "BMIN3", "GUAR3", "EKTR4", "TASA3", "PSSA3",
    "CURY3", "CPLE6", "BMKS3", "GMAT3", "IGTI3", "BPAC11", "BPAN4", "ANIM3", "EPAR3", "FLRY3", "KLBN4", "MELK3",
    "MNPR3", "KLBN11", "ENMT4", "CPLE5", "HBTS5", "MGLU3", "KLBN3", "CEAB3", "PORT3", "ALOS3", "VAMO3", "FESA3",
    "EQTL3", "TEND3", "NATU3", "SOJA3", "BMOB3", "PNVL3", "ENMT3", "LREN3", "B3SA3", "ABEV3", "PGMN3", "IGTI11",
    "MDIA3", "STBP3", "GGPS3", "AFLT3", "ASAI3", "BAUH4", "EMAE4", "CBAV3", "LIPR3", "TFCO4", "DXCO3", "MOTV3",
    "VIVT3", "SEER3", "RDOR3", "AZZA3", "BSLI3", "ELET5", "BRST3", "BSLI4", "RENT3", "JPSA3", "FRAS3", "HYPE3",
    "RADL3", "PEAB4", "BPAC3", "PEAB3", "UCAS3", "EMBR3", "ALPA3", "SMFT3", "WEGE3", "ALPA4", "TOTS3", "GEPA3",
    "GEPA4", "USIM5", "USIM3", "LOGN3", "ELMD3", "REAG3", "EVEN3", "ESPA3", "ENEV3", "BRAV3", "GPAR3", "LWSA3",
    "PRNR3", "MNDL3", "USIM6", "LAND3", "TELB4", "QUAL3", "ORVR3", "TIMS3", "POSI3", "TELB3", "SRNA3", "AURE3"
]

# Adiciona .SA no final de cada ticker
ativos_brasil_sa = [ticker + '.SA' for ticker in ativos_brasil]

# Ativos padrão
default_brasil = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'B3SA3.SA']

# Multiselect atualizado
ativos = st.multiselect("Selecione os ativos:", ativos_brasil_sa, default=default_brasil)

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
    beta = covar[0, 1] / covar[1, 1]
    treynor = retorno_esperado / beta
    var_95 = np.percentile(ret_port, 5) * np.sqrt(252)
    cvar_95 = ret_port[ret_port <= np.percentile(ret_port, 5)].mean() * np.sqrt(252)
    acumulado = (1 + ret_port).cumprod()
    drawdown = acumulado / acumulado.cummax() - 1
    max_dd = drawdown.min()

    st.markdown("## Carteira Ótima (Sharpe Máximo)")

    # Indicadores lado a lado com destaque
    st.markdown("""
        <style>
        .indicador {
            font-size: 20px;
            padding: 10px;
            margin: 5px;
            background-color: #f3f3f3;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='indicador'><strong>Retorno Esperado</strong><br>{retorno_esperado:.2%}</div>
        <div class='indicador'><strong>Volatilidade</strong><br>{volatilidade:.2%}</div>
        <div class='indicador'><strong>Sharpe</strong><br>{sharpe_ratio:.2f}</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='indicador'><strong>Sortino</strong><br>{sortino:.2f}</div>
        <div class='indicador'><strong>Treynor</strong><br>{treynor:.2f}</div>
        <div class='indicador'><strong>Beta</strong><br>{beta:.2f}</div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='indicador'><strong>VaR (95%)</strong><br>{var_95:.2%}</div>
        <div class='indicador'><strong>CVaR (95%)</strong><br>{cvar_95:.2%}</div>
        <div class='indicador'><strong>Max Drawdown</strong><br>{max_dd:.2%}</div>
        """, unsafe_allow_html=True)

    # Fronteira
    df_fronteira = pd.DataFrame({'Retorno': rets, 'Risco': riscos, 'Sharpe': sharpe})
    fig_fronteira = px.scatter(df_fronteira, x='Risco', y='Retorno', color='Sharpe',
                               title="Fronteira Eficiente", color_continuous_scale='Viridis')
    fig_fronteira.add_trace(go.Scatter(x=[riscos[idx_sharpe_max]], y=[rets[idx_sharpe_max]],
                                       mode='markers', marker=dict(color='red', size=12),
                                       name='Máx. Sharpe'))
    st.plotly_chart(fig_fronteira, use_container_width=True)

    # Desempenho Acumulado
    ibov = benchmark.loc[ret_port.index]
    base100_port = (1 + ret_port).cumprod() * 100
    base100_ibov = (1 + ibov).cumprod() * 100
    fig_acum = go.Figure()
    fig_acum.add_trace(go.Scatter(y=base100_port, name="Carteira"))
    fig_acum.add_trace(go.Scatter(y=base100_ibov, name="Ibovespa"))
    fig_acum.update_layout(title="Desempenho Acumulado (Base 100)")
    st.plotly_chart(fig_acum, use_container_width=True)

    # Volatilidade Móvel
    vol_port = ret_port.rolling(30).std() * np.sqrt(252)
    vol_ibov = ibov.rolling(30).std() * np.sqrt(252)
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(y=vol_port, name="Carteira"))
    fig_vol.add_trace(go.Scatter(y=vol_ibov, name="Ibovespa"))
    fig_vol.update_layout(title="Volatilidade Móvel (30 dias)")
    st.plotly_chart(fig_vol, use_container_width=True)

    # Drawdown
    dd_ibov = (1 + ibov).cumprod()
    dd_ibov = dd_ibov / dd_ibov.cummax() - 1
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(y=drawdown, name="Carteira"))
    fig_dd.add_trace(go.Scatter(y=dd_ibov, name="Ibovespa"))
    fig_dd.update_layout(title="Drawdown Comparado")
    st.plotly_chart(fig_dd, use_container_width=True)

    # Pesos
    st.markdown("### Alocação da Carteira Ótima")
    fig_pesos = px.bar(x=ativos, y=melhor_pesos, labels={'x': 'Ativo', 'y': 'Peso'},
                       title="Distribuição de Pesos na Carteira")
    st.plotly_chart(fig_pesos, use_container_width=True)

    # Correlação
    st.markdown("### Matriz de Correlação dos Ativos")
    corr = retornos.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                         title="Matriz de Correlação")
    st.plotly_chart(fig_corr, use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown("<center>Desenvolvido pelo <strong>Prof. Luiz Eduardo Gaio</strong> para fins educacionais</center>", unsafe_allow_html=True)

