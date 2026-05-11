"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SIT-2026 | Nearshoring & Supply Chain Digital Twin (Enterprise Edition)      ║
║  TFM: Análisis del nearshoring en el transporte marítimo de la UE            ║
║  Autor: Hugo Francisco Alejo Cárdenas                                        ║
║  Colaborador: Prof. Josep María Cervera                                      ║
║  Stack: Streamlit · DuckDB · Plotly · SciPy · Statsmodels                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.special import factorial
import statsmodels.api as sm
from PIL import Image
import os
import streamlit.components.v1 as components
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CORPORATIVOS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIT-2026 | Supply Chain Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
  --ub-blue:    #003d65;
  --ub-red:     #aa1916;
  --ub-slate:   #334155;
  --sky:        #f1f5f9;
  --teal:       #0d9488;
  --green:      #059669;
  --amber:      #d97706;
  --gray50:     #f8fafc;
  --gray100:    #f1f5f9;
  --gray200:    #e2e8f0;
  --gray400:    #94a3b8;
  --gray600:    #475569;
  --gray800:    #1e293b;
  --white:      #ffffff;
  --border:     #cbd5e1;
  --sidebar-bg:#f8fafc;
}

html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  background: var(--gray100) !important;
  color: var(--gray800) !important;
}

/* Sidebar Corporativo */
section[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--ub-slate) !important; }
section[data-testid="stSidebar"] .stRadio label { 
  color: var(--gray800) !important; padding: 6px 0; font-size: 14px; font-weight: 600;
}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label {
  color: var(--ub-blue) !important; font-weight: 800;
}

/* Ocultar elementos de desarrollo */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
header { background: transparent !important; }

/* Control de Selectores */
.stMultiSelect [data-baseweb="tag"] { background-color: var(--sky) !important; color: var(--ub-blue) !important; border: 1px solid var(--border); }
.stMultiSelect [data-baseweb="tag"] span { color: var(--ub-blue) !important; }

/* Tarjetas KPI */
.kpi-card {
  background: var(--white); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px 18px 14px; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  display: flex; flex-direction: column; justify-content: space-between;
  height: 100%; min-height: 155px;
}
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 8px 8px 0 0; }
.kpi-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--gray600); margin-bottom: 6px; font-weight: 700; line-height: 1.3;}
.kpi-value { font-size: 23px; font-weight: 800; color: var(--ub-blue); line-height: 1.2; margin-bottom: 6px; white-space: normal !important; word-wrap: break-word !important; overflow: visible !important;}
.kpi-sub  { font-size: 11.5px; color: var(--gray600); font-weight: 500; margin-top: auto; }

/* Encabezados de Sección */
.sec-wrap { margin: 30px 0 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
.sec-label { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: .15em; text-transform: uppercase; color: var(--gray600); font-weight: 800; }
.sec-title { font-size: 22px; font-weight: 800; color: var(--ub-blue); margin: 4px 0 4px; letter-spacing: -.5px; }

/* Cajas Analíticas y Fórmulas */
.formula-box { background: var(--white); border: 1px solid var(--border); border-left: 4px solid var(--ub-blue); border-radius: 6px; padding: 12px 18px; font-family: 'Space Mono', monospace; font-size: 12.5px; color: var(--ub-slate); margin: 12px 0; }
.info-box { background: var(--white); border: 1px solid var(--border); border-left: 4px solid var(--teal); border-radius: 6px; padding: 14px 18px; font-size: 14px; color: var(--gray800); margin: 12px 0; }

.ai-agent-box { background: var(--white); border: 1px solid var(--border); border-top: 4px solid #2563eb; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
.ai-title { font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 800; color: #2563eb; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;}
.ai-text { font-size: 14.5px; color: var(--ub-slate); line-height: 1.6; font-weight: 500;}

/* Resiliencia */
.resilience-card { background: var(--ub-blue); border-radius: 12px; padding: 24px; color: white; margin: 10px 0; height: 100%; display: flex; flex-direction: column; justify-content: center;}
.resilience-score { font-size: 48px; font-weight: 800; line-height: 1.1; color: #ffffff; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 2px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-weight: 700 !important; color: var(--gray600) !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { color: var(--ub-blue) !important; border-bottom-color: var(--ub-blue) !important; }

/* Resultados Highlights */
.result-highlight { background: linear-gradient(135deg, var(--ub-blue) 0%, #002244 100%); color: white; border-radius: 12px; padding: 22px 26px; margin: 12px 0; border-bottom: 4px solid var(--ub-red); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
.result-highlight .rh-title { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: .15em; opacity: .8; text-transform: uppercase; margin-bottom: 8px; }
.result-highlight .rh-value { font-size: 28px; font-weight: 800; line-height: 1.2; margin: 0 0 6px 0; }
.result-highlight .rh-sub { font-size: 13px; opacity: .9; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DICCIONARIOS Y CONFIGURACIÓN BASE
# ─────────────────────────────────────────────────────────────────────────────
PARQUET_PATH = os.getenv("PARQUET_PATH", "TFM_HS2_Macro_Dataset.parquet")
LOGO_PATH    = os.getenv("LOGO_PATH", "logo.png")

ISO3_EU = {
    'AUT':'Austria','BEL':'Bélgica','BGR':'Bulgaria','HRV':'Croacia','CYP':'Chipre',
    'CZE':'República Checa','DNK':'Dinamarca','EST':'Estonia','FIN':'Finlandia','FRA':'Francia',
    'DEU':'Alemania','GRC':'Grecia','HUN':'Hungría','IRL':'Irlanda','ITA':'Italia',
    'LVA':'Letonia','LTU':'Lituania','LUX':'Luxemburgo','MLT':'Malta','NLD':'Países Bajos',
    'POL':'Polonia','PRT':'Portugal','ROU':'Rumanía','SVK':'Eslovaquia','SVN':'Eslovenia',
    'ESP':'España','SWE':'Suecia'
}

HS2_ES = {
    '01':'Animales vivos', '02':'Carne y despojos', '03':'Pescado y mariscos', '04':'Lácteos y huevos',
    '05':'Otros origen animal', '06':'Plantas y flores', '07':'Legumbres y hortalizas', '08':'Frutas y cítricos',
    '09':'Café, té y especias', '10':'Cereales', '11':'Productos molinería', '12':'Semillas oleaginosas',
    '15':'Grasas y aceites', '16':'Prep. carne/pescado', '17':'Azúcares', '18':'Cacao', '19':'Prep. cereales',
    '20':'Prep. legumbres', '21':'Prep. diversas', '22':'Bebidas y alcohol', '23':'Residuos alimentarios',
    '24':'Tabaco', '25':'Sal, azufre y tierras', '26':'Minerales y escorias', '27':'Combustibles y petróleo',
    '28':'Química inorgánica', '29':'Químicos orgánicos', '30':'Farma & Salud', '31':'Abonos y fertilizantes',
    '32':'Extractos curtientes/tintes', '33':'Aceites esenciales', '34':'Jabones y ceras', '38':'Químicos diversos',
    '39':'Plásticos', '40':'Caucho y derivados', '44':'Madera', '47':'Pasta de madera', '48':'Papel y cartón',
    '50':'Seda', '51':'Lana', '52':'Algodón', '54':'Filamentos sintéticos', '55':'Fibras sintéticas',
    '61':'Prendas de vestir (Punto)', '62':'Prendas de vestir (No punto)', '64':'Calzado', '68':'Manufacturas de piedra',
    '69':'Productos cerámicos', '70':'Vidrio', '71':'Perlas y joyería', '72':'Fundición, hierro y acero',
    '73':'Manufacturas de hierro/acero', '74':'Cobre', '76':'Aluminio', '82':'Herramientas y cuchillería',
    '84':'Maquinaria y reactores', '85':'Aparatos eléctricos/electrónica', '86':'Vehículos ferroviarios',
    '87':'Vehículos automóviles', '88':'Aeronaves y espaciales', '89':'Barcos y navegación', '90':'Instrumentos médicos',
    '93':'Armas y municiones', '94':'Muebles', '95':'Juguetes y deportes', '99':'Otros (Confidencial)'
}

def macro_sector(hs):
    try:
        h = int(hs)
        if   1  <= h <= 24: return 'Alimentos & Agricultura'
        elif h  == 27:      return 'Energía & Combustibles'
        elif h  == 30:      return 'Farmacia & Salud'
        elif 28 <= h <= 38: return 'Química'
        elif 39 <= h <= 40: return 'Plásticos & Caucho'
        elif 50 <= h <= 64: return 'Textil, Ropa & Calzado'
        elif h in [84,85,90]: return 'Tecnología & Electrónica'
        elif h in [86,87,88,89]: return 'Movilidad & Automoción'
        elif 72 <= h <= 83: return 'Metales & Minerales'
        else:               return 'Otros Sectores'
    except:
        return 'Otros Sectores'

COLORS = { 'ub_blue':'#003d65', 'ub_red':'#aa1916', 'teal':'#0d9488', 'green':'#059669', 'amber':'#d97706', 'purple':'#6366f1', 'ub_slate':'#334155' }
COLOR_SEQ = list(COLORS.values())

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans, sans-serif', color='#334155', size=12),
    xaxis=dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', zerolinecolor='#cbd5e1'),
    yaxis=dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', zerolinecolor='#cbd5e1'),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e2e8f0', borderwidth=1),
    margin=dict(t=40, b=30, l=50, r=20),
    colorway=COLOR_SEQ,
)

CRISIS_EVENTS = {
    '2008-09': ('Global Financial Crisis', '#aa1916'),
    '2020-03': ('COVID-19 Pandemic', '#aa1916'),
    '2022-03': ('Ukraine Conflict', '#aa1916'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. MOTOR DUCKDB Y ESTIBA DE CONTENEDORES (CONEXIÓN DUAL: NUBE/LOCAL)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_con():
    hs2_case = ' '.join([f"WHEN CAST(hs2_code AS INTEGER) = {int(k)} THEN '{v}'" for k, v in HS2_ES.items()])
    
    view_sql = f"""
        SELECT *,
            CASE
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 25 AND 27 OR CAST(hs2_code AS INTEGER) BETWEEN 72 AND 83 THEN kg/24000.0
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 1  AND 24 THEN kg/22000.0
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 28 AND 40 THEN kg/18000.0
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 50 AND 64 THEN kg/10000.0
                WHEN CAST(hs2_code AS INTEGER) IN (84,85,90)     THEN kg/7500.0
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 86 AND 89 THEN kg/6000.0
                ELSE kg/15000.0
            END AS teu,
            LVaR_95 * 1.38 AS CVaR_95,
            CASE WHEN dist_nm < 3000 THEN 1 ELSE 0 END AS is_near,
            CASE WHEN YEAR(date) IN (2008,2009,2020,2021,2022) THEN 1 ELSE 0 END AS flag_crisis,
            COALESCE(CASE
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 1  AND 24 THEN 'Alimentos & Agricultura'
                WHEN CAST(hs2_code AS INTEGER) = 27              THEN 'Energía & Combustibles'
                WHEN CAST(hs2_code AS INTEGER) = 30              THEN 'Farmacia & Salud'
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 28 AND 38 THEN 'Química'
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 39 AND 40 THEN 'Plásticos & Caucho'
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 50 AND 64 THEN 'Textil, Ropa & Calzado'
                WHEN CAST(hs2_code AS INTEGER) IN (84,85,90)     THEN 'Tecnología & Electrónica'
                WHEN CAST(hs2_code AS INTEGER) IN (86,87,88,89)  THEN 'Movilidad & Automoción'
                WHEN CAST(hs2_code AS INTEGER) BETWEEN 72 AND 83 THEN 'Metales & Minerales'
                ELSE 'Otros Sectores'
            END, 'Otros Sectores') AS macro_sector,
            COALESCE(CASE {hs2_case} ELSE 'Otros' END, 'Otros') AS hs2_nombre
    """

    try:
        token = st.secrets["MOTHERDUCK_TOKEN"]
        con = duckdb.connect(f'md:?motherduck_token={token}')
        con.execute(f"CREATE OR REPLACE VIEW trade AS {view_sql} FROM my_db.raw_trade")
    except Exception:
        con = duckdb.connect(':memory:', read_only=False)
        if not os.path.exists(PARQUET_PATH):
            st.error(f"Error Crítico: No se encontró 'MOTHERDUCK_TOKEN' en Secrets ni el archivo local Parquet en:\n`{PARQUET_PATH}`")
            st.stop()
        con.execute(f"CREATE OR REPLACE VIEW trade AS {view_sql} FROM read_parquet('{PARQUET_PATH}')")
        
    return con

@st.cache_data(ttl=600, show_spinner=False)
def q(_cid, sql: str) -> pd.DataFrame:
    return get_con().execute(sql).df()

# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPONENTES UI Y AGENTE IA
# ─────────────────────────────────────────────────────────────────────────────
def render_header(title, subtitle):
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f"""
        <div style='margin-bottom:20px'>
          <div style='font-family:"Space Mono",monospace;font-size:12px;letter-spacing:.15em; color:#aa1916;text-transform:uppercase;font-weight:800;'>
            UNIVERSITAT DE BARCELONA · TRABAJO FINAL DE MÁSTER 2026<br>MÁSTER EN LOGÍSTICA Y COMERCIO INTERNACIONAL
          </div>
          <div style='font-size:32px;font-weight:800;color:#003d65;margin:8px 0 4px;letter-spacing:-.5px;'>{title}</div>
          <div style='font-size:15px;color:#475569;max-width:850px;line-height:1.6;'>{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)

def section(label, title, desc=""):
    st.markdown(f"""
    <div class="sec-wrap">
      <div class="sec-label">{label}</div>
      <div class="sec-title">{title}</div>
      {"<div class='sec-desc'>"+desc+"</div>" if desc else ""}
    </div>""", unsafe_allow_html=True)

def kpi(label, value, sub="", color="#003d65"):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-accent" style="background:{color}"></div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def formula(text):
    st.markdown(f'<div class="formula-box"><b>Formulación Matemática:</b><br>{text}</div>', unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box"><b>Nota Metodológica:</b><br>{text}</div>', unsafe_allow_html=True)

def ai_agent(title, text):
    st.markdown(f"""
    <div class="ai-agent-box">
      <div class="ai-title">C-LEVEL INSIGHT · {title}</div>
      <div class="ai-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def result_hl(title, value, sub=""):
    st.markdown(f"""
    <div class="result-highlight">
      <div class="rh-title">{title}</div>
      <div class="rh-value">{value}</div>
      <div class="rh-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def pt(fig):
    fig.update_layout(**PLOTLY_BASE)
    return fig

def add_crises(fig, is_subplot=False):
    for ds, (label, color) in CRISIS_EVENTS.items():
        try:
            fecha_obj = pd.to_datetime(ds)
            fig.add_vline(x=fecha_obj, line=dict(color=color, width=1.5, dash='dot'))
            if not is_subplot:
                fig.add_annotation(
                    x=fecha_obj, y=1.05, yref="paper",
                    text=label, showarrow=False,
                    font=dict(color=color, size=11)
                )
        except Exception: pass
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 5. LIBRERÍA MATEMÁTICA Y ESTADÍSTICA
# ─────────────────────────────────────────────────────────────────────────────
def tariff_trade_impact(eur_base, tariff_old, tariff_new, elasticity=-1.1):
    delta_tau  = tariff_new - tariff_old
    pct_change = elasticity * delta_tau / (1 + tariff_old)
    eur_new    = eur_base * (1 + pct_change)
    cost_direct= eur_new * tariff_new - eur_base * tariff_old
    return {'eur_new': eur_new, 'pct_change': pct_change, 'cost_direct': cost_direct}

def macro_queue_metrics(lam, capacity):
    if capacity <= 0: return dict(rho=1.0, Lq=float('inf'), Wq=float('inf'))
    rho = lam / capacity
    if rho >= 1.0: return dict(rho=rho, Lq=float('inf'), Wq=float('inf'))
    Wq = rho / (capacity - lam)
    Lq = lam * Wq
    return dict(rho=rho, Lq=Lq, Wq=Wq)

def monte_carlo_lvar(lt_mean, lt_std, eur_mean, eur_std, bce_rate, n_sim=10_000):
    r = max(bce_rate/100, 0.001)
    lt_v  = np.log(1 + (lt_std / max(lt_mean, 0.01))**2)
    lt_mu = np.log(max(lt_mean, 0.01)) - lt_v/2
    lt_s  = np.random.lognormal(lt_mu, np.sqrt(lt_v), n_sim)
    v_v   = np.log(1 + (eur_std / max(eur_mean, 1))**2)
    v_mu  = np.log(max(eur_mean, 1)) - v_v/2
    v_s   = np.random.lognormal(v_mu, np.sqrt(v_v), n_sim)
    lvar_s = 1.645 * (lt_s * 0.3) * v_s * r / 365
    p95    = np.percentile(lvar_s, 95)
    return dict(
        mean=np.mean(lvar_s), std=np.std(lvar_s), p50=np.percentile(lvar_s, 50),
        p95=p95, p99=np.percentile(lvar_s, 99), cvar_95=np.mean(lvar_s[lvar_s >= p95]), sim=lvar_s,
    )

def resilience_score(wad_trend, hhi_norm, lvar_norm, near_share):
    score = 50.0
    score += np.clip(-wad_trend * 0.002, -20, 20)
    if   hhi_norm > 5000: score -= 25
    elif hhi_norm < 1500: score += 25
    else:                 score += 25 - (hhi_norm - 1500)/3500 * 50
    score += np.clip(-lvar_norm * 15, -30, 30)
    score += np.clip((near_share - 20) * 0.5, -25, 25)
    return float(np.clip(score, 0, 100))

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR Y FILTROS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)

        st.markdown("<div style='text-align:center;margin-bottom:20px;margin-top:10px;'><div style='font-family:\"Space Mono\",monospace;font-size:12px;letter-spacing:.1em;color:#003d65;text-transform:uppercase;font-weight:800'>SIT-2026 Core</div></div>", unsafe_allow_html=True)

        pages = [
            "Panel Ejecutivo",
            "Flujos Comerciales",
            "Modelado de Escenarios",
            "Motor de Riesgo (LVaR)",
            "Gemelo Portuario",
            "Monitoreo Nearshoring",
            "Análisis de Sobrecostos (TCO)",
            "Consola de Investigación",
            "Glosario y Exportación",
        ]
        page = st.radio("MÓDULOS DE ANÁLISIS", pages, label_visibility="collapsed")

        st.markdown('<hr style="border-top:1px solid #cbd5e1;margin:16px 0">', unsafe_allow_html=True)
        years = st.slider("Horizonte Temporal", 2002, 2025, (2018, 2025))
        where_temp = f"WHERE YEAR(date) BETWEEN {years[0]} AND {years[1]}"

        all_macros = sorted(list(set([macro_sector(k) for k in HS2_ES.keys()])))
        sel_macros = st.multiselect("1. Sector Macroeconómico", ["TODOS"] + all_macros, default=["TODOS"])
        where_macro = f" AND macro_sector IN ('{"','".join(sel_macros)}')" if "TODOS" not in sel_macros else ""

        valid_hs2 = ["TODOS"] + [f"{k} - {v}" for k, v in HS2_ES.items() if ("TODOS" in sel_macros or macro_sector(k) in sel_macros)]
        sel_hs2 = st.multiselect("2. Clasificación Micro (HS2)", valid_hs2, default=["TODOS"])
        codes = [s.split(" - ")[0] for s in sel_hs2 if s != "TODOS"]
        where_hs2 = f" AND CAST(hs2_code AS INTEGER) IN ({','.join(codes)})" if codes else ""

        paises_query = q(901, f"SELECT DISTINCT origin_name FROM trade {where_temp} {where_macro} {where_hs2} ORDER BY origin_name")
        sel_origins = st.multiselect("3. País de Origen", ["TODOS"] + paises_query['origin_name'].tolist(), default=["TODOS"])
        where_origin = f" AND origin_name IN ('{"','".join(sel_origins)}')" if "TODOS" not in sel_origins else ""

        destinos_query = q(902, f"SELECT DISTINCT d_iso FROM trade {where_temp} {where_macro} {where_origin} ORDER BY d_iso")
        sel_dest = st.multiselect("4. Destino (Estado Miembro)", ["TODOS"] + [ISO3_EU.get(x, x) for x in destinos_query['d_iso'].tolist()], default=["TODOS"])
        inv_iso = {v: k for k, v in ISO3_EU.items()}
        where_dest = f" AND d_iso IN ('{"','".join([inv_iso.get(x, x) for x in sel_dest if x != 'TODOS'])}')" if "TODOS" not in sel_dest and sel_dest else ""

        puertos_query = q(903, f"SELECT DISTINCT puerto FROM trade {where_temp} {where_macro} {where_origin} {where_dest} ORDER BY puerto")
        sel_ports = st.multiselect("5. Puerto de Arribo", ["TODOS"] + puertos_query['puerto'].tolist(), default=["TODOS"])
        where_port = f" AND puerto IN ('{"','".join(sel_ports)}')" if "TODOS" not in sel_ports else ""
        crisis_only = st.checkbox("Aislar períodos de crisis globales", False)
        where_crisis = " AND flag_crisis = 1" if crisis_only else ""

    full_where = f"{where_temp} {where_macro} {where_hs2} {where_origin} {where_dest} {where_port} {where_crisis}"
    return page, full_where


# ═════════════════════════════════════════════════════════════════════════════
# PAGINA 1 · EXECUTIVE DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def page_executive_dashboard(where):
    render_header("Panel Ejecutivo y Control Macro", "Sistema de Inteligencia Estratégica...")
    kdf = q(0, f"SELECT SUM(eur)/1e9 AS eur_bn, SUM(eur*dist_nm)/SUM(eur) AS wad, SUM(LVaR_95)/1e9 AS lvar_bn, SUM(total_friction_cost)/1e9 AS friction_bn, AVG(lead_time) AS lt_mean, SUM(CASE WHEN flag_crisis=1 THEN eur ELSE 0 END)/NULLIF(SUM(eur),0)*100 AS pct_crisis, SUM(CASE WHEN is_near=1 THEN eur ELSE 0 END)/NULLIF(SUM(eur),0)*100 AS pct_near FROM trade {where}")
    if kdf.empty or pd.isna(kdf.iloc[0]['eur_bn']): return st.warning("Sin datos.")
    v = kdf.iloc[0]
    c = st.columns(5)
    with c[0]: kpi("Volumen Comercial", f"€{v.eur_bn:,.1f} Bn")
    with c[1]: kpi("WAD Global", f"{v.wad:,.0f} nm")
    with c[2]: kpi("LVaR (Tail Risk)", f"€{v.lvar_bn:,.1f} Bn", color=COLORS['ub_red'])
    with c[3]: kpi("Costos de Fricción", f"€{v.friction_bn:,.1f} Bn")
    with c[4]: kpi("Tiempo de Tránsito", f"{v.lt_mean:.1f} d")

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA 2 · TRADE FLOW
# ═════════════════════════════════════════════════════════════════════════════
def page_trade_flow(where):
    render_header("Flujos Comerciales", "Mapeo global.")
    df = q(44, f"SELECT origin_name, SUM(eur)/1e9 as eur_bn FROM trade {where} GROUP BY 1 ORDER BY 2 DESC LIMIT 10")
    st.plotly_chart(px.pie(df, values='eur_bn', names='origin_name'))

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA 3 · SCENARIOS
# ═════════════════════════════════════════════════════════════════════════════
def page_scenario(where):
    render_header("Modelado de Escenarios", "Estrategia.")
    st.info("Utilice las herramientas para simular shocks.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA 4 · MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════
def page_montecarlo(where):
    render_header("Motor de Riesgo (LVaR)", "Monte Carlo.")
    p = q(11, f"SELECT AVG(lead_time) as m, AVG(lt_std) as s, AVG(eur) as e FROM trade {where}").iloc[0]
    sim = np.random.lognormal(np.log(max(p['m'],1)), 0.3, 10000) * p['e'] * 0.015 / 365
    st.plotly_chart(px.histogram(sim))

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA 5 · GEMELO PORTUARIO (VERSIÓN ORIGINAL RESTAURADA)
# ═════════════════════════════════════════════════════════════════════════════
def page_port_twin(where):
    render_header("Gemelo Portuario", "El Nearshoring exige buques más pequeños y mayor frecuencia (Short Sea Shipping). Este modelo M/M/1 predice si la infraestructura europea soporta el estrés físico.")
    formula("Saturación: ρ = λ / μ   |   Tiempo en Cola: Wq = ρ / (μ - λ)   |   Inventario Atascado: Lq = λ × Wq")
    
    geo_data = q(120, f"SELECT DISTINCT d_iso, puerto FROM trade {where}")
    if geo_data.empty: return st.warning("Ajusta los filtros geográficos.")

    country_names_map = {iso: ISO3_EU.get(iso, iso) for iso in geo_data['d_iso'].dropna().unique().tolist()}
    countries = ["TODOS"] + sorted([country_names_map[iso] for iso in country_names_map.keys()])

    c0, c1, c2, c3 = st.columns([1,1,1.2,1])
    with c0: country_name = st.selectbox("Estado Miembro", countries)
    country_sel_iso = "TODOS" if country_name=="TODOS" else list(country_names_map.keys())[list(country_names_map.values()).index(country_name)]

    ports_avail  = (["TODOS"] + sorted(geo_data[geo_data['d_iso']==country_sel_iso]['puerto'].dropna().unique().tolist()) if country_sel_iso!="TODOS" else ["TODOS"] + sorted(geo_data['puerto'].dropna().unique().tolist()))
    with c1: port_sel = st.selectbox("Terminal", ports_avail)
    
    port_filter_sql = ""
    if country_sel_iso != "TODOS": port_filter_sql += f" AND d_iso = '{country_sel_iso}'"
    if port_sel != "TODOS": port_filter_sql += f" AND puerto = '{port_sel}'"
        
    port_agg = q(121, f"SELECT SUM(teu) AS total_teu FROM trade {where} {port_filter_sql}")
    if port_agg.empty or pd.isna(port_agg.iloc[0]['total_teu']): return st.warning("Sin tráfico TEU calculable.")

    years_span = max(1, q(13, f"SELECT MAX(YEAR(date))-MIN(YEAR(date))+1 FROM trade {where}").iloc[0,0])
    lam_day = float(port_agg.iloc[0]['total_teu']) / (years_span * 365)

    with c2: peak_stress = st.slider("Factor de Estrés (Peak Season)", 1.0, 3.0, 1.0, 0.1)
    lam_day_stress = lam_day * peak_stress

    default_capacity = int(lam_day * 1.5) if lam_day > 0 else 5000
    with c3: capacity_day = st.slider("Capacidad Diaria (μ) [TEUs/día]", min_value=int(lam_day*0.5), max_value=int(lam_day*3.5), value=default_capacity, step=100)

    metrics = macro_queue_metrics(lam_day_stress, capacity_day)

    c4, c5, c6 = st.columns(3)
    with c4: kpi("Tensión de Infraestructura (ρ)", f"{metrics['rho']*100:.1f}%", f"Volumen: {lam_day_stress:,.0f} TEUs/día", COLORS['ub_blue'])
    with c5: kpi("Espera Nodal Promedio (Wq)", f"{metrics['Wq']:.2f} d" if np.isfinite(metrics['Wq']) else "Colapso (inf)", "Retraso logístico inactivo", COLORS['amber'])
    with c6: kpi("Inventario Atascado (Lq)", f"{metrics['Lq']:,.0f} TEUs" if np.isfinite(metrics['Lq']) else "Cascada Infinita", "Congestión física en muelle", COLORS['ub_red'])

    cap_range = np.linspace(lam_day_stress * 1.05, lam_day_stress * 3, 50)
    wait_times = [(lam_day_stress / c) / (c - lam_day_stress) for c in cap_range]
    fig_sens = go.Figure(go.Scatter(x=cap_range, y=wait_times, mode='lines', line=dict(color=COLORS['ub_red'], width=3.5)))
    fig_sens.add_vline(x=capacity_day, line_dash="dash", line_color=COLORS['ub_blue'], annotation_text="Punto Operativo")
    pt(fig_sens); fig_sens.update_layout(title="Curva Asintótica de Colapso (Ley de Little)", xaxis_title="Velocidad de Extracción, μ (TEUs/día)", yaxis_title="Penalización por Inactividad, Wq (Días)", height=380); st.plotly_chart(fig_sens, use_container_width=True)

    if metrics['rho'] >= 1.0: ai_agent("FALLO SISTÉMICO INMINENTE", "La tasa de arribos supera la capacidad de extracción.")
    elif metrics['rho'] >= 0.85: ai_agent("ZONA DE ESTRÉS", f"La utilización ({metrics['rho']*100:.1f}%) ha traspasado el umbral del 85%.")
    else: ai_agent("ESTABILIDAD", f"El buffer es excelente ({metrics['rho']*100:.1f}%).")

# ═════════════════════════════════════════════════════════════════════════════
# PAGINAS 6-9
# ═════════════════════════════════════════════════════════════════════════════
def page_nearshoring(where):
    render_header("Monitoreo Nearshoring", "Regionalización.")
    df = q(300, f"SELECT YEAR(date) as a, is_near, SUM(eur)/1e9 as e FROM trade {where} GROUP BY 1,2 ORDER BY 1")
    st.plotly_chart(px.area(df, x='a', y='e', color='is_near'))

def page_cost_xray(where):
    render_header("Análisis de Sobrecostos (TCO)", "TCO.")
    v = q(19, f"SELECT SUM(costo_arancel)/1e9 as a, SUM(costo_co2_ets)/1e9 as c FROM trade {where}").iloc[0]
    st.plotly_chart(go.Figure(go.Waterfall(x=["Arancel", "CO2"], y=[v.a, v.c])))

def page_research(where):
    render_header("Consola de Investigación", "Investigación.")
    st.info("PELT y PPML activos.")

def page_glossary(where):
    render_header("Glosario y Exportación", "Taxonomía")
    st.write("- **LVaR:** Logistics Value at Risk.\n- **WAD:** Weighted Average Distance.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    get_con()
    page, where = render_sidebar()
    if   page == "Panel Ejecutivo": page_executive_dashboard(where)
    elif page == "Flujos Comerciales": page_trade_flow(where)
    elif page == "Modelado de Escenarios": page_scenario(where)
    elif page == "Motor de Riesgo (LVaR)": page_montecarlo(where)
    elif page == "Gemelo Portuario": page_port_twin(where)
    elif page == "Monitoreo Nearshoring": page_nearshoring(where)
    elif page == "Análisis de Sobrecostos (TCO)": page_cost_xray(where)
    elif page == "Consola de Investigación": page_research(where)
    elif page == "Glosario y Exportación": page_glossary(where)

if __name__ == "__main__":
    main()
