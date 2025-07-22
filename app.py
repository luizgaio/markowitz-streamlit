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

col1, col2, col3 = st.columns(3)
with col1:
    data_inicio = st.date_input("Data de início", value=datetime.today() - timedelta(days=3*365))
with col2:
    data_fim = st.date_input("Data de fim", value=datetime.today())
with col3:
    taxa_rf = st.number_input("Taxa Livre de Risco (a.a.)", value=0.00, format="%.2f") / 100

# Coleta de dados
if len(ativos) >= 2:
    dados = yf.download(ativos + ['^BVSP'], start=data_inicio, end=data_fim)['Close'].dropna()
    retornos = np.log(dados[ativos] / dados[ativos].shift(1)).dropna()
    benchmark = np.log(dados['^BVSP'] / dados['^BVSP'].shift(1)).dropna()

    media = retornos.mean() * 252
    cov = retornos.cov() * 252

    # Simulação de carteiras
    n = 5000
    simul_pesos = np.random.dirichlet(np.ones(len(ativos)), n)
    rets = simul_pesos @ media.values
    riscos = np.sqrt(np.einsum('ij,jk,ik->i', simul_pesos, cov.values, simul_pesos))
    sharpe = (rets - taxa_rf) / riscos

    # Escolha da carteira
    opcao_carteira = st.selectbox(
        "Escolha a carteira a ser analisada:",
        ["Carteira Própria", "Máximo Sharpe", "Máximo Sortino", "Máximo Treynor"]
    )

    if opcao_carteira == "Carteira Própria":
        st.markdown("Insira os pesos (soma deve ser 1.0):")
        pesos_input = []
        for i, ativo in enumerate(ativos):
            peso = st.number_input(f"Peso para {ativo}", min_value=0.0, max_value=1.0, step=0.01, key=f"peso_{i}")
            pesos_input.append(peso)
        pesos = np.array(pesos_input)

        if not np.isclose(np.sum(pesos), 1.0):
            st.warning("A soma dos pesos deve ser 1.0")
            st.stop()

    elif opcao_carteira == "Máximo Sharpe":
        idx_otimo = np.argmax(sharpe)
        pesos = simul_pesos[idx_otimo]

    elif opcao_carteira == "Máximo Sortino":
        downside = np.where(retornos < 0, retornos, 0)
        downside_cov = downside.cov() * 252
        downside_risco = np.sqrt(np.einsum('ij,jk,ik->i', simul_pesos, downside_cov.values, simul_pesos))
        sortino = (rets - taxa_rf) / downside_risco
        idx_otimo = np.argmax(sortino)
        pesos = simul_pesos[idx_otimo]

    elif opcao_carteira == "Máximo Treynor":
        betas = np.array([np.cov(retornos[ativo], benchmark.loc[retornos.index])[0, 1] / np.var(benchmark.loc[retornos.index]) for ativo in ativos])
        beta_portf = simul_pesos @ betas
        treynor = (rets - taxa_rf) / beta_portf
        idx_otimo = np.argmax(treynor)
        pesos = simul_pesos[idx_otimo]

    # Cálculo dos retornos da carteira escolhida
    ret_port = retornos @ pesos

    # Exibição do retorno acumulado
    base100 = (dados[ativos] / dados[ativos].iloc[0]) * 100
    fig_base = px.line(base100, title="Preços dos Ativos - Base 100")
    st.plotly_chart(fig_base, use_container_width=True)

    # Cálculo de indicadores
    retorno_esperado = ret_port.mean() * 252
    volatilidade = ret_port.std() * np.sqrt(252)
    sharpe_ratio = (retorno_esperado - taxa_rf) / volatilidade
    downside = ret_port[ret_port < 0].std() * np.sqrt(252)
    sortino_ratio = (retorno_esperado - taxa_rf) / downside
    covar = np.cov(ret_port, benchmark.loc[ret_port.index])
    beta = covar[0, 1] / covar[1, 1]
    treynor_ratio = (retorno_esperado - taxa_rf) / beta
    var_95 = np.percentile(ret_port, 5) * np.sqrt(252)
    cvar_95 = ret_port[ret_port <= np.percentile(ret_port, 5)].mean() * np.sqrt(252)
    acumulado = (1 + ret_port).cumprod()
    drawdown = acumulado / acumulado.cummax() - 1
    max_dd = drawdown.min()

    # Indicadores visuais
    st.markdown("## Carteira Ótima")
    col1, col2, col3 = st.columns(3)
    col1.metric("Retorno Esperado", f"{retorno_esperado:.2%}")
    col1.metric("Volatilidade", f"{volatilidade:.2%}")
    col1.metric("Sharpe", f"{sharpe_ratio:.2f}")

    col2.metric("Sortino", f"{sortino_ratio:.2f}")
    col2.metric("Treynor", f"{treynor_ratio:.2f}")
    col2.metric("Beta", f"{beta:.2f}")

    col3.metric("VaR (95%)", f"{var_95:.2%}")
    col3.metric("CVaR (95%)", f"{cvar_95:.2%}")
    col3.metric("Máx Drawdown", f"{max_dd:.2%}")

    # Tabela de pesos
    df_pesos = pd.DataFrame({"Ativo": ativos, "Peso": pesos})
    df_pesos["Peso"] = df_pesos["Peso"].apply(lambda x: f"{x:.2%}")
    st.dataframe(df_pesos, use_container_width=True)

    # Rodapé
    st.markdown("---")
    st.markdown("<center>Desenvolvido pelo <strong>Prof. Luiz Eduardo Gaio</strong> para fins educacionais</center>", unsafe_allow_html=True)
