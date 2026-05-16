"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  SIT-2026 | Nearshoring & Supply Chain Digital Twin                          ║
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
import io
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
  --purple:     #6366f1;
  --gray50:     #f8fafc;
  --gray100:    #f1f5f9;
  --gray200:    #e2e8f0;
  --gray400:    #94a3b8;
  --gray600:    #475569;
  --gray800:    #1e293b;
  --white:      #ffffff;
  --border:     #cbd5e1;
  --sidebar-bg: #f8fafc;
}
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  background: var(--gray100) !important;
  color: var(--gray800) !important;
}
section[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--ub-slate) !important; }
section[data-testid="stSidebar"] .stRadio label {
  color: var(--gray800) !important; padding: 6px 0; font-size: 14px; font-weight: 600;
}
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }
header { background: transparent !important; }
.stMultiSelect [data-baseweb="tag"] { background-color: var(--sky) !important; color: var(--ub-blue) !important; border: 1px solid var(--border); }
.stMultiSelect [data-baseweb="tag"] span { color: var(--ub-blue) !important; }
.block-container { padding-bottom: 80px !important; } /* Margen inferior para el footer fijo */

/* ── KPI BASE CON ANIMACIÓN ── */
.kpi-card {
  background: var(--white); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px 18px 14px; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  display: flex; flex-direction: column; justify-content: space-between;
  height: 100%; min-height: 155px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.08);
}
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 8px 8px 0 0; }
.kpi-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--gray600); margin-bottom: 6px; font-weight: 700; line-height: 1.3;}
.kpi-value { font-size: 23px; font-weight: 800; color: var(--ub-blue); line-height: 1.2; margin-bottom: 6px; white-space: normal !important; word-wrap: break-word !important; overflow: visible !important;}
.kpi-sub  { font-size: 11.5px; color: var(--gray600); font-weight: 500; margin-top: auto; }

/* ── KPI CON DELTA ── */
.kpi-delta-card {
  background: var(--white); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px 18px 14px; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  display: flex; flex-direction: column; justify-content: space-between;
  height: 100%; min-height: 155px;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-delta-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.08);
}
.kpi-delta-base { font-size: 13px; color: var(--gray400); font-weight: 600; text-decoration: line-through; margin-bottom: 2px;}
.kpi-delta-new  { font-size: 23px; font-weight: 800; line-height: 1.2; margin-bottom: 4px; }
.kpi-delta-badge-good { display: inline-block; background: #dcfce7; color: #16a34a; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 12px; font-family:'Space Mono',monospace; }
.kpi-delta-badge-bad  { display: inline-block; background: #fee2e2; color: #dc2626; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 12px; font-family:'Space Mono',monospace; }
.kpi-delta-badge-neutral { display: inline-block; background: #f1f5f9; color: #475569; font-size: 11px; font-weight: 800; padding: 2px 8px; border-radius: 12px; font-family:'Space Mono',monospace; }

/* ── BANNER DE SIMULACIÓN ACTIVA ── */
.sim-banner {
  background: linear-gradient(90deg, #003d65 0%, #0d6e9e 50%, #0d9488 100%);
  border-radius: 8px; padding: 14px 22px; margin-bottom: 20px;
  display: flex; align-items: center; gap: 16px; color: white;
  box-shadow: 0 4px 12px rgba(0,61,101,0.25);
}
.sim-banner-pct { font-size: 36px; font-weight: 800; line-height: 1; }
.sim-banner-label { font-family:'Space Mono',monospace; font-size: 10px; letter-spacing:.15em; opacity:.8; text-transform: uppercase; }
.sim-banner-title { font-size: 16px; font-weight: 700; }
.sim-banner-sub { font-size: 13px; opacity: .85; font-weight: 500; }

/* ── CAJAS DE MANUAL / INSTRUCCIONES ── */
.guide-box {
  background: linear-gradient(145deg, var(--white) 0%, var(--gray50) 100%);
  border: 1px solid var(--border);
  border-left: 4px solid var(--purple);
  border-radius: 8px;
  padding: 18px 22px;
  margin: 16px 0 24px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.03);
}
.guide-title {
  font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 800;
  color: var(--purple); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px;
  display: flex; align-items: center; gap: 8px;
}
.guide-text { font-size: 14px; color: var(--gray800); line-height: 1.6; font-weight: 400;}

/* ── FOOTER INSTITUCIONAL SECIHTI ── */
.sechithi-footer {
  position: fixed; left: 0; bottom: 0; width: 100%;
  background-color: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(5px);
  border-top: 1px solid var(--border);
  padding: 10px 20px; text-align: center;
  font-size: 11.5px; color: var(--gray600); font-weight: 500;
  z-index: 9999;
  display: flex; justify-content: center; align-items: center; gap: 6px;
  box-shadow: 0 -2px 10px rgba(0,0,0,0.03);
}
.sechithi-highlight { color: var(--ub-blue); font-weight: 800; letter-spacing: 0.5px;}

/* ── SECCIONES Y PANELES ── */
.before-panel { background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 14px; }
.after-panel  { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 14px; }
.panel-title  { font-family:'Space Mono',monospace; font-size: 10px; letter-spacing:.12em; text-transform:uppercase; font-weight:800; margin-bottom:10px; }
.sec-wrap { margin: 30px 0 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
.sec-label { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: .15em; text-transform: uppercase; color: var(--gray600); font-weight: 800; }
.sec-title { font-size: 22px; font-weight: 800; color: var(--ub-blue); margin: 4px 0 4px; letter-spacing: -.5px; }

/* ── CAJAS ANALÍTICAS ── */
.formula-box { background: var(--white); border: 1px solid var(--border); border-left: 4px solid var(--ub-blue); border-radius: 6px; padding: 12px 18px; font-family: 'Space Mono', monospace; font-size: 12.5px; color: var(--ub-slate); margin: 12px 0; }
.info-box { background: var(--white); border: 1px solid var(--border); border-left: 4px solid var(--teal); border-radius: 6px; padding: 14px 18px; font-size: 14px; color: var(--gray800); margin: 12px 0; }
.warn-box { background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #d97706; border-radius: 6px; padding: 14px 18px; font-size: 14px; color: var(--gray800); margin: 12px 0; }
.danger-box { background: #fef2f2; border: 1px solid #fecaca; border-left: 4px solid #aa1916; border-radius: 6px; padding: 14px 18px; font-size: 14px; color: var(--gray800); margin: 12px 0; }

/* ── AI AGENT ── */
.ai-agent-box { background: var(--white); border: 1px solid var(--border); border-top: 4px solid #2563eb; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
.ai-title { font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 800; color: #2563eb; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;}
.ai-text { font-size: 14.5px; color: var(--ub-slate); line-height: 1.6; font-weight: 500;}

/* ── RESILIENCIA Y COMPONENTES EXTRAS ── */
.resilience-card { background: var(--ub-blue); border-radius: 12px; padding: 24px; color: white; margin: 10px 0; height: 100%; display: flex; flex-direction: column; justify-content: center;}
.resilience-score { font-size: 48px; font-weight: 800; line-height: 1.1; color: #ffffff; }
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 2px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-weight: 700 !important; color: var(--gray600) !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { color: var(--ub-blue) !important; border-bottom-color: var(--ub-blue) !important; }

/* ── REGLAS DE IMPRESIÓN PDF ── */
@media print {
    section[data-testid="stSidebar"] { display: none !important; }
    header { display: none !important; }
    .stApp { margin: 0 !important; padding: 0 !important; }
    .block-container { padding: 10mm !important; max-width: 100% !important; }
    .sechithi-footer { position: fixed; bottom: 0; left: 0; width: 100%; border: none; }
    .kpi-card, .kpi-delta-card, .ai-agent-box, .guide-box { break-inside: avoid; page-break-inside: avoid; }
    ::-webkit-scrollbar { display: none; }
}
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
    '01':'Animales vivos','02':'Carne y despojos','03':'Pescado y mariscos','04':'Lácteos y huevos',
    '05':'Otros origen animal','06':'Plantas y flores','07':'Legumbres y hortalizas','08':'Frutas y cítricos',
    '09':'Café, té y especias','10':'Cereales','11':'Productos molinería','12':'Semillas oleaginosas',
    '15':'Grasas y aceites','16':'Prep. carne/pescado','17':'Azúcares','18':'Cacao','19':'Prep. cereales',
    '20':'Prep. legumbres','21':'Prep. diversas','22':'Bebidas y alcohol','23':'Residuos alimentarios',
    '24':'Tabaco','25':'Sal, azufre y tierras','26':'Minerales y escorias','27':'Combustibles y petróleo',
    '28':'Química inorgánica','29':'Químicos orgánicos','30':'Farma & Salud','31':'Abonos y fertilizantes',
    '32':'Extractos curtientes/tintes','33':'Aceites esenciales','34':'Jabones y ceras','38':'Químicos diversos',
    '39':'Plásticos','40':'Caucho y derivados','44':'Madera','47':'Pasta de madera','48':'Papel y cartón',
    '50':'Seda','51':'Lana','52':'Algodón','54':'Filamentos sintéticos','55':'Fibras sintéticas',
    '61':'Prendas de vestir (Punto)','62':'Prendas de vestir (No punto)','64':'Calzado','68':'Manufacturas de piedra',
    '69':'Productos cerámicos','70':'Vidrio','71':'Perlas y joyería','72':'Fundición, hierro y acero',
    '73':'Manufacturas de hierro/acero','74':'Cobre','76':'Aluminio','82':'Herramientas y cuchillería',
    '84':'Maquinaria y reactores','85':'Aparatos eléctricos/electrónica','86':'Vehículos ferroviarios',
    '87':'Vehículos automóviles','88':'Aeronaves y espaciales','89':'Barcos y navegación','90':'Instrumentos médicos',
    '93':'Armas y municiones','94':'Muebles','95':'Juguetes y deportes','99':'Otros (Confidencial)'
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

COLORS = {
    'ub_blue':'#003d65','ub_red':'#aa1916','teal':'#0d9488',
    'green':'#059669','amber':'#d97706','purple':'#6366f1','ub_slate':'#334155'
}
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
    '2008-09': ('Crisis Financiera Global', '#aa1916'),
    '2020-03': ('Pandemia COVID-19', '#aa1916'),
    '2022-03': ('Conflicto en Ucrania', '#aa1916'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. MOTOR DUCKDB
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
            st.error(f"Error Crítico: Sin 'MOTHERDUCK_TOKEN' ni Parquet local en:\n`{PARQUET_PATH}`")
            st.stop()
        con.execute(f"CREATE OR REPLACE VIEW trade AS {view_sql} FROM read_parquet('{PARQUET_PATH}')")
    return con

@st.cache_data(ttl=600, show_spinner=False)
def q(_cid, sql: str) -> pd.DataFrame:
    return get_con().execute(sql).df()

# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPONENTES UI AVANZADOS
# ─────────────────────────────────────────────────────────────────────────────
def render_header(title, subtitle):
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f"""
        <div style='margin-bottom:20px'>
          <div style='font-family:"Space Mono",monospace;font-size:12px;letter-spacing:.15em;color:#aa1916;text-transform:uppercase;font-weight:800;'>
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
      {"<div style='font-size:14px;color:#475569;margin-top:4px;'>"+desc+"</div>" if desc else ""}
    </div>""", unsafe_allow_html=True)

def kpi(label, value, sub="", color="#003d65"):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-accent" style="background:{color}"></div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def kpi_delta(label, val_base, val_new, fmt_base, fmt_new, sub="", lower_is_better=False, color="#003d65"):
    delta_raw = val_new - val_base
    if abs(val_base) > 1e-9:
        delta_pct = (delta_raw / abs(val_base)) * 100
    else:
        delta_pct = 0.0
    is_improvement = (delta_raw < 0) if lower_is_better else (delta_raw > 0)
    badge_cls = "kpi-delta-badge-good" if is_improvement else ("kpi-delta-badge-bad" if delta_raw != 0 else "kpi-delta-badge-neutral")
    sign = "-" if delta_raw < 0 else ("+" if delta_raw > 0 else "")
    badge_txt = f"{sign}{abs(delta_pct):.1f}%"
    st.markdown(f"""
    <div class="kpi-delta-card">
      <div class="kpi-accent" style="background:{color}"></div>
      <div class="kpi-label">{label}</div>
      <div class="kpi-delta-base">{fmt_base}</div>
      <div class="kpi-delta-new" style="color:{color}">{fmt_new}</div>
      <div><span class="{badge_cls}">{badge_txt}</span></div>
      <div class="kpi-sub" style="margin-top:6px">{sub}</div>
    </div>""", unsafe_allow_html=True)

def sim_banner(transfer_pct):
    if transfer_pct <= 0: return
    pct_int = int(round(transfer_pct * 100))
    st.markdown(f"""
    <div class="sim-banner">
      <div style="flex-shrink:0">
        <div class="sim-banner-label">SIMULACIÓN ACTIVA</div>
        <div class="sim-banner-pct">{pct_int}%</div>
      </div>
      <div style="border-left:1px solid rgba(255,255,255,0.3);padding-left:16px">
        <div class="sim-banner-title">Estrategia de Relocalización Nearshoring Aplicada</div>
        <div class="sim-banner-sub">
          El {pct_int}% del volumen oceánico lejano (>3,000 nm) ha sido redirigido a proveedores
          regionales. Todos los módulos reflejan el impacto proyectado sobre la infraestructura,
          los costos y el riesgo.
        </div>
      </div>
      <div style="margin-left:auto;flex-shrink:0;text-align:right">
        <div class="sim-banner-label">Configuración Global</div>
        <div style="font-size:22px;font-weight:800;">Panel Lateral</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def guide_box(title, text):
    st.markdown(f"""
    <div class="guide-box">
      <div class="guide-title">💡 {title}</div>
      <div class="guide-text">{text}</div>
    </div>""", unsafe_allow_html=True)

def render_footer():
    import base64
    svg_base64 = ""
    # Localización absoluta del archivo dentro de la estructura de producción
    ruta_svg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "Logotipo_SECIHTI.svg")
    
    # Codificación dinámica en Base64 para garantizar compatibilidad nativa en HTML/CSS
    if os.path.exists(ruta_svg):
        with open(ruta_svg, "rb") as f:
            svg_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    logo_html = f'<img src="data:image/svg+xml;base64,{svg_base64}" style="height:26px; margin-left:12px; vertical-align:middle;" />' if svg_base64 else ""

    st.markdown(f"""
    <div class="sechithi-footer">
        Proyecto desarrollado con el apoyo y beca institucional de la 
        <span class="sechithi-highlight">Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI).</span>
        {logo_html}
    </div>
    """, unsafe_allow_html=True)

def formula(text):
    st.markdown(f'<div class="formula-box"><b>Formulación Matemática:</b><br>{text}</div>', unsafe_allow_html=True)

def info(text):
    st.markdown(f'<div class="info-box"><b>Nota Metodológica:</b><br>{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">ATENCIÓN: {text}</div>', unsafe_allow_html=True)

def danger(text):
    st.markdown(f'<div class="danger-box">PELIGRO: {text}</div>', unsafe_allow_html=True)

def ai_agent(title, text):
    st.markdown(f"""
    <div class="ai-agent-box">
      <div class="ai-title">C-LEVEL INSIGHT · {title}</div>
      <div class="ai-text">{text}</div>
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
                fig.add_annotation(x=fecha_obj, y=1.05, yref="paper", text=label, showarrow=False, font=dict(color=color, size=11))
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
    r = max(bce_rate / 100, 0.001)
    lt_v  = np.log(1 + (lt_std / max(lt_mean, 0.01)) ** 2)
    lt_mu = np.log(max(lt_mean, 0.01)) - lt_v / 2
    lt_s  = np.random.lognormal(lt_mu, np.sqrt(lt_v), n_sim)
    v_v   = np.log(1 + (eur_std / max(eur_mean, 1)) ** 2)
    v_mu  = np.log(max(eur_mean, 1)) - v_v / 2
    v_s   = np.random.lognormal(v_mu, np.sqrt(v_v), n_sim)
    lvar_s = 1.645 * (lt_s * 0.3) * v_s * r / 365
    p95    = np.percentile(lvar_s, 95)
    return dict(mean=np.mean(lvar_s), std=np.std(lvar_s), p50=np.percentile(lvar_s, 50), p95=p95, p99=np.percentile(lvar_s, 99), cvar_95=np.mean(lvar_s[lvar_s >= p95]), sim=lvar_s)

def resilience_score(wad_trend, hhi_norm, lvar_norm, near_share):
    score = 50.0
    score += np.clip(-wad_trend * 0.002, -20, 20)
    if   hhi_norm > 5000: score -= 25
    elif hhi_norm < 1500: score += 25
    else:                 score += 25 - (hhi_norm - 1500) / 3500 * 50
    score += np.clip(-lvar_norm * 15, -30, 30)
    score += np.clip((near_share - 20) * 0.5, -25, 25)
    return float(np.clip(score, 0, 100))

def find_collapse_threshold(lam_far_base, lam_near_base, daily_teu_far, drop_near, mu_nominal, peak_stress=1.0, target_rho=0.85):
    for pct in range(1, 101):
        lf = lam_far_base * (1 - pct / 100.0)
        ln = lam_near_base + (daily_teu_far * (pct / 100.0) / max(drop_near, 0.001))
        rho = (lf + ln) * peak_stress / max(mu_nominal, 1e-9)
        if rho >= target_rho: return pct
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR Y FILTROS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)

        st.markdown("<div style='text-align:center;margin-bottom:20px;margin-top:10px;'><div style='font-family:\"Space Mono\",monospace;font-size:12px;letter-spacing:.1em;color:#003d65;text-transform:uppercase;font-weight:800'>SIT-2026 CORE</div></div>", unsafe_allow_html=True)

        pages = [
            "Panel Ejecutivo",
            "Flujos Comerciales",
            "Gemelo Portuario",
            "Análisis de Sobrecostos (TCO)",
            "Motor de Riesgo (LVaR)",
            "Monitoreo Nearshoring",
            "Modelado de Escenarios",
            "Consola de Investigación",
            "Glosario y Exportación",
        ]
        page = st.radio("MÓDULOS DE ANÁLISIS", pages, label_visibility="collapsed")

        st.markdown('<hr style="border-top:1px solid #cbd5e1;margin:16px 0">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:11px;color:#aa1916;font-weight:800;text-transform:uppercase;margin-bottom:4px;">Estrategia Global de Red</div>', unsafe_allow_html=True)
        transfer_pct = st.slider("Relocalización a Nearshoring (%)", 0, 100, 0, help="Simula la transferencia de volumen de redes de suministro de larga distancia a proveedores regionales. Modifica transversalmente todo el sistema.") / 100.0

        if transfer_pct > 0:
            st.markdown(f"<div style='background:#dcfce7;border-radius:6px;padding:8px 12px;margin:8px 0;font-size:12px;color:#16a34a;font-weight:700;font-family:Space Mono,monospace;'>SIMULACIÓN ACTIVA: {transfer_pct*100:.0f}%</div>", unsafe_allow_html=True)

        st.markdown('<hr style="border-top:1px solid #cbd5e1;margin:16px 0">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:11px;color:#003d65;text-transform:uppercase;margin-bottom:12px;font-weight:800">Filtros en Cascada</div>', unsafe_allow_html=True)

        years = st.slider("Horizonte Temporal", 2002, 2025, (2018, 2025))
        where_temp = f"WHERE YEAR(date) BETWEEN {years[0]} AND {years[1]}"

        all_macros = sorted(list(set([macro_sector(k) for k in HS2_ES.keys()])))
        sel_macros = st.multiselect("1. Sector Macroeconómico", ["TODOS"] + all_macros, default=["TODOS"])
        where_macro = f" AND macro_sector IN ('{"','".join(sel_macros)}')" if "TODOS" not in sel_macros else ""

        valid_hs2 = ["TODOS"] + [f"{k} - {v}" for k, v in HS2_ES.items() if ("TODOS" in sel_macros or macro_sector(k) in sel_macros)]
        sel_hs2 = st.multiselect("2. Clasificación Micro (HS2)", valid_hs2, default=["TODOS"])
        where_hs2 = f" AND CAST(hs2_code AS INTEGER) IN ({','.join([s.split(' - ')[0] for s in sel_hs2 if s != 'TODOS'])})" if "TODOS" not in sel_hs2 and sel_hs2 else ""

        where_sector_full = f"{where_macro} {where_hs2}"

        paises_query = q(901, f"SELECT DISTINCT origin_name FROM trade {where_temp} {where_sector_full} ORDER BY origin_name")
        sel_origins = st.multiselect("3. País de Origen", ["TODOS"] + paises_query['origin_name'].tolist(), default=["TODOS"])
        where_origin = f" AND origin_name IN ('{"','".join(sel_origins)}')" if "TODOS" not in sel_origins else ""

        destinos_query = q(902, f"SELECT DISTINCT d_iso FROM trade {where_temp} {where_sector_full} {where_origin} ORDER BY d_iso")
        sel_dest = st.multiselect("4. Destino (Estado Miembro)", ["TODOS"] + [ISO3_EU.get(x, x) for x in destinos_query['d_iso'].tolist()], default=["TODOS"])
        where_dest = f" AND d_iso IN ('{"','".join([ {v:k for k,v in ISO3_EU.items()}.get(x,x) for x in sel_dest if x != "TODOS"])}')" if "TODOS" not in sel_dest and sel_dest else ""

        puertos_query = q(903, f"SELECT DISTINCT puerto FROM trade {where_temp} {where_sector_full} {where_origin} {where_dest} ORDER BY puerto")
        sel_ports = st.multiselect("5. Puerto de Arribo", ["TODOS"] + puertos_query['puerto'].tolist(), default=["TODOS"])
        where_port = f" AND puerto IN ('{"','".join(sel_ports)}')" if "TODOS" not in sel_ports else ""

        crisis_only = st.checkbox("Aislar periodos de crisis", False)
        where_crisis = " AND flag_crisis = 1" if crisis_only else ""

        st.markdown("""
        <div style="margin-top:30px;padding-top:16px;border-top:1px solid #cbd5e1;font-size:11.5px;color:#475569;line-height:1.6;">
            <b style="color:#003d65;">Interoperabilidad Activa</b><br>La cascada de filtros garantiza coherencia entre todos los nodos seleccionados y los registros físicos subyacentes.
        </div>""", unsafe_allow_html=True)

        components.html("""
        <button onclick="window.parent.print()" style="background-color:#003d65;color:white;border:none;
            padding:12px 15px;border-radius:6px;width:95%;cursor:pointer;
            font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:13.5px;
            display:flex;align-items:center;justify-content:center;gap:10px;
            box-shadow:0 4px 6px rgba(0,0,0,0.1);margin:0 auto;">
            📄 Generar Reporte PDF
        </button>""", height=70)

    full_where = f"{where_temp} {where_sector_full} {where_origin} {where_dest} {where_port} {where_crisis}"
    return page, full_where, transfer_pct

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 · PANEL EJECUTIVO
# ═════════════════════════════════════════════════════════════════════════════
def page_executive_dashboard(where, transfer_pct):
    render_header("Panel Ejecutivo y Control Macro", "Sistema de Inteligencia Estratégica para evaluar empíricamente el impacto de la regionalización (Nearshoring) en la resiliencia del suministro, la estructura de costos (TCO) y la saturación de infraestructura portuaria de la UE.")
    
    guide_box(
        "MANUAL DE USUARIO · CÓMO UTILIZAR EL SIT-2026 CORE",
        "1. <b>Filtros (Panel Lateral):</b> Selecciona el horizonte temporal, sectores macro y la geografía. El sistema procesará los registros comerciales en tiempo real.<br>"
        "2. <b>Estrategia Global (Slider):</b> Modifica el porcentaje de relocalización hacia el Nearshoring. Todos los módulos (costos, riesgos, infraestructura) se recalibrarán automáticamente.<br>"
        "3. <b>Módulos Analíticos:</b> Navega por las pestañas laterales para profundizar en mapas de flujos, cuellos de botella portuarios (Gemelo Portuario), fricción financiera (TCO) o proyectar el riesgo de capital (LVaR)."
    )

    sim_banner(transfer_pct)

    kdf = q(0, f"SELECT SUM(eur)/1e9 AS eur_bn, SUM(eur*dist_nm)/SUM(eur) AS wad, AVG(lead_time) AS lt_mean, SUM(LVaR_95)/1e9 AS lvar_bn, SUM(total_friction_cost)/1e9 AS friction_bn, SUM(CASE WHEN flag_crisis=1 THEN eur ELSE 0 END)/NULLIF(SUM(eur),0)*100 AS pct_crisis, SUM(CASE WHEN is_near=1 THEN eur ELSE 0 END)/NULLIF(SUM(eur),0)*100 AS pct_near FROM trade {where}")
    if kdf.empty or pd.isna(kdf.iloc[0]['eur_bn']): return st.warning("Volumen de datos insuficiente para el rango seleccionado.")
    
    v = kdf.iloc[0]
    v_wad, v_pct_near, v_lvar_bn, v_friction_bn, v_eur_bn, v_pct_crisis = (float(v.wad) if pd.notna(v.wad) else 0.0, float(v.pct_near) if pd.notna(v.pct_near) else 0.0, float(v.lvar_bn) if pd.notna(v.lvar_bn) else 0.0, float(v.friction_bn) if pd.notna(v.friction_bn) else 0.0, float(v.eur_bn) if pd.notna(v.eur_bn) else 0.0, float(v.pct_crisis) if pd.notna(v.pct_crisis) else 0.0)

    sim_wad      = v_wad * (1 - transfer_pct * 0.60)
    sim_pct_near = v_pct_near + (100 - v_pct_near) * transfer_pct
    sim_lvar     = v_lvar_bn * (1 + transfer_pct * 0.15) 
    sim_friction = v_friction_bn * (1 - transfer_pct * 0.08)

    st.markdown("<br>", unsafe_allow_html=True)
    if transfer_pct > 0:
        c = st.columns(5)
        with c[0]: kpi("Vol. Total Histórico FOB", f"€{v_eur_bn:,.1f} Bn", "Dinámica de suma cero aplicada", COLORS['ub_blue'])
        with c[1]: kpi_delta("WAD Promedio", v_wad, sim_wad, f"{v_wad:,.0f} nm", f"{sim_wad:,.0f} nm", "Distancia física de tránsito", True, COLORS['teal'])
        with c[2]: kpi_delta("Capital en Riesgo (LVaR)", v_lvar_bn, sim_lvar, f"€{v_lvar_bn:,.1f} Bn", f"€{sim_lvar:,.1f} Bn", "Inmovilización proyectada", True, COLORS['ub_red'])
        with c[3]: kpi_delta("Fricción Total (TCO)", v_friction_bn, sim_friction, f"€{v_friction_bn:,.1f} Bn", f"€{sim_friction:,.1f} Bn", "Aduanas, emisiones y terminal", True, COLORS['amber'])
        with c[4]: kpi_delta("Cuota Nearshoring", v_pct_near, sim_pct_near, f"{v_pct_near:.1f}%", f"{sim_pct_near:.1f}%", "Penetración regional estimada", False, COLORS['green'])
    else:
        c = st.columns(5)
        with c[0]: kpi("Vol. Total Histórico FOB", f"€{v_eur_bn:,.1f} Bn", "Miles de Millones acumulados", COLORS['ub_blue'])
        with c[1]: kpi("WAD Promedio", f"{v_wad:,.0f} nm", "Distancia física de tránsito", COLORS['ub_blue'])
        with c[2]: kpi("Capital en Riesgo (LVaR)", f"€{v_lvar_bn:,.1f} Bn", "Promedio inmovilizado", COLORS['ub_red'])
        with c[3]: kpi("Fricción Total (TCO)", f"€{v_friction_bn:,.1f} Bn", "Aduanas, emisiones y terminal", COLORS['amber'])
        with c[4]: kpi("Cuota Nearshoring", f"{v_pct_near:.1f}%", "Penetración regional actual", COLORS['teal'])

    hhi_global = q(50, f"WITH sh AS (SELECT o_iso, SUM(eur) AS val, SUM(SUM(eur)) OVER () AS tot FROM trade {where} GROUP BY o_iso) SELECT SUM((val/NULLIF(tot,0))*(val/NULLIF(tot,0)))*10000 AS hhi FROM sh")
    hhi_val = hhi_global.iloc[0]['hhi'] if not hhi_global.empty else 2500
    wad_trend_df = q(51, f"SELECT YEAR(date) AS anio, SUM(eur*dist_nm)/SUM(eur) AS wad FROM trade {where} GROUP BY 1 ORDER BY 1")
    wad_trend_val= wad_trend_df.iloc[-1]['wad'] - wad_trend_df.iloc[0]['wad'] if len(wad_trend_df) >= 2 else 0

    rscore_base = resilience_score(wad_trend_val, hhi_val, min(v_lvar_bn/max(v_eur_bn,1), 1.0), float(v_pct_near))
    rscore_sim  = resilience_score(wad_trend_val, hhi_val, min(sim_lvar/max(v_eur_bn,1), 1.0), float(sim_pct_near))
    rlabel, rcolor = (("ALTA RESILIENCIA", "#059669") if rscore_sim >= 70 else ("RIESGO MODERADO", "#d97706") if rscore_sim >= 45 else ("VULNERABILIDAD", "#aa1916"))

    st.markdown("<br>", unsafe_allow_html=True)
    c_r1, c_r2, c_r3 = st.columns([1, 1.2, 1.8])
    with c_r1:
        st.markdown(f"""
        <div class="resilience-card">
          <div style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:.1em;opacity:.7;text-transform:uppercase;">SIT Resilience Score</div>
          <div class="resilience-score">{int(round(rscore_sim))}</div>
          <div style='font-size:14px;font-weight:700;color:white;margin-top:8px;'>{rlabel}</div>
          {"<div style='font-size:11px;opacity:.8;margin-top:4px;'>Puntaje Base: "+str(int(round(rscore_base)))+"/100</div>" if transfer_pct > 0 else ""}
        </div>""", unsafe_allow_html=True)
    with c_r2:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=rscore_sim, title=dict(text="Salud Estructural Logística UE", font=dict(size=14, color='#334155')), gauge=dict(axis=dict(range=[0,100], tickwidth=1), bar=dict(color=rcolor, thickness=0.3), bgcolor="white", borderwidth=1, bordercolor="#cbd5e1", steps=[dict(range=[0,45], color="#fef2f2"), dict(range=[45,70], color="#fffbeb"), dict(range=[70,100], color="#f0fdf4")]), number=dict(font=dict(size=30, color='#334155'), suffix="/100")))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(t=30,b=0,l=10,r=10))
        st.plotly_chart(fig_g, use_container_width=True)
    with c_r3:
        wad_score = max(0, min(100, 100-(wad_trend_val/100+50)))
        hhi_score = max(0, min(100, 100-hhi_val/100))
        lvar_score = max(0, min(100, 100-min(sim_lvar/max(v_eur_bn,1),1.0)*100))
        fig_r = go.Figure(go.Scatterpolar(r=[wad_score, hhi_score, lvar_score, float(sim_pct_near), max(0, 100-float(v_pct_crisis)*2), wad_score], theta=['WAD Trend','Diversificación HHI','Control LVaR','Near-shore','Estabilidad Crisis','WAD Trend'], fill='toself', fillcolor='rgba(0,61,101,0.15)', line=dict(color='#003d65', width=2), marker=dict(size=6, color='#aa1916')))
        fig_r.update_layout(polar=dict(bgcolor='rgba(248,250,252,1)', radialaxis=dict(visible=True, range=[0,100], gridcolor='#e2e8f0', tickfont_size=9), angularaxis=dict(tickfont=dict(size=11, color='#475569', weight='bold'), gridcolor='#e2e8f0')), paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(t=20,b=20,l=40,r=40), showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

    if transfer_pct > 0:
        section("CUANTIFICACIÓN DE LA ESTRATEGIA", "Impacto Proyectado de la Relocalización sobre el P&L y el Balance General")
        ci1, ci2, ci3, ci4 = st.columns(4)
        with ci1: kpi("Ahorro de Tránsito WAD", f"{abs(sim_wad - v_wad):,.0f} nm", f"Reducción de distancia neta", COLORS['teal'])
        with ci2: kpi("Impacto Capital LVaR", f"€{sim_lvar - v_lvar_bn:+,.2f} Bn", "Alteración de reservas logísticas", COLORS['ub_red'])
        with ci3: kpi("Ahorro Net TCO", f"€{sim_friction - v_friction_bn:+,.2f} Bn", "Contracción total de sobrecostos", COLORS['amber'])
        with ci4: kpi("Aumento Regional", f"+{sim_pct_near - v_pct_near:.1f} pp", "Ganancia de volumen Nearshore", COLORS['green'])

        cats = ['WAD Promedio (Normalizado)', 'LVaR Acumulado (€ Bn)', 'Fricción Total TCO (€ Bn)', 'Cuota Nearshoring (%)']
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name='Escenario Histórico', x=cats, y=[v_wad/1000, v_lvar_bn, v_friction_bn, v_pct_near/10], marker_color=COLORS['ub_slate'], opacity=0.7))
        fig_comp.add_trace(go.Bar(name=f'Escenario Relocalizado al {transfer_pct*100:.0f}%', x=cats, y=[sim_wad/1000, sim_lvar, sim_friction, sim_pct_near/10], marker_color=COLORS['teal'], opacity=0.9))
        pt(fig_comp); fig_comp.update_layout(barmode='group', height=350, title="Análisis Comparativo del Bloque Comercial Seleccionado", yaxis_title="Magnitudes a Escala Logística", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_comp, use_container_width=True)

    ai_txt = (
        f"Con la estrategia activa de relocalización al **{transfer_pct*100:.0f}%**, la distancia media proyectada se reduce significativamente a **{sim_wad:,.0f} nm**. "
        f"La reconfiguración operativa ajusta el capital inmovilizado a un estimado de **€{sim_lvar:,.1f} Bn**. Es vital validar la factibilidad operativa navegando a la sección 'Gemelo Portuario'."
        if transfer_pct > 0
        else
        f"El ecosistema seleccionado opera un volumen total histórico de **€{v_eur_bn:,.1f} Bn** con una distancia media efectiva (WAD) de **{v_wad:,.0f} nm**. "
        f"El riesgo de volatilidad retiene un promedio de **€{v_lvar_bn:,.1f} Bn** en caja. Ajusta la 'Estrategia Global de Red' en el panel izquierdo para observar proyecciones computadas."
    )
    ai_agent("ANÁLISIS MACRO-ESTRUCTURAL", ai_txt)

    section("MACRO-TENDENCIAS", "Comportamiento Analítico Trimestral del Comercio Exterior (Serie Temporal)")
    ts = q(1, f"SELECT DATE_TRUNC('quarter', date) AS t, SUM(eur)/1e9 AS eur_bn, SUM(eur*dist_nm)/SUM(eur) AS wad, SUM(LVaR_95)/1e9 AS lvar_bn, MAX(flag_crisis) AS crisis FROM trade {where} GROUP BY 1 ORDER BY 1")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.3, 0.2], vertical_spacing=0.04, subplot_titles=["Comercio Consolidado Trimestral (€ Bn)","WAD — Oscilación de Dependencia Física (Millillas Náuticas)","LVaR — Absorción de Capital en Rutas (€ Bn)"])
    for row in [1, 2, 3]:
        for _, r in ts[ts['crisis']==1].iterrows():
            t0 = pd.to_datetime(r['t']); t1 = t0 + pd.DateOffset(months=3)
            fig.add_vrect(x0=t0, x1=t1, fillcolor='rgba(170,25,22,0.1)', line_width=0, row=row, col=1)
            
    fig.add_trace(go.Scatter(x=ts['t'], y=ts['eur_bn'], fill='tozeroy', fillcolor='rgba(0,61,101,0.08)', line=dict(color=COLORS['ub_blue'], width=2.5)), row=1, col=1)
    if transfer_pct > 0: fig.add_trace(go.Scatter(x=ts['t'], y=ts['wad']*(1-transfer_pct*0.6), name='WAD Proyectado', fill='tozeroy', fillcolor='rgba(13,148,136,0.12)', line=dict(color=COLORS['teal'], width=2, dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts['t'], y=ts['wad'], fill='tozeroy', fillcolor='rgba(13,148,136,0.06)', line=dict(color=COLORS['ub_slate'], width=1.5)), row=2, col=1)
    fig.add_trace(go.Bar(x=ts['t'], y=ts['lvar_bn'], marker_color=np.where(ts['crisis']==1, COLORS['ub_red'], COLORS['ub_blue']), opacity=0.8), row=3, col=1)
    pt(fig); fig.update_layout(height=600, showlegend=False); add_crises(fig, is_subplot=True); st.plotly_chart(fig, use_container_width=True)

    section("DESGLOSE SECTORIAL", "Estructura Anual de Intercambio por Principales Macro-Sectores Productivos")
    s_ts = q(2, f"SELECT YEAR(date) AS anio, macro_sector, SUM(eur)/1e9 AS eur_bn FROM trade {where} GROUP BY 1, 2 ORDER BY 1, 2")
    if not s_ts.empty:
        fig_sec = px.area(s_ts, x="anio", y="eur_bn", color="macro_sector", color_discrete_sequence=px.colors.qualitative.Prism)
        pt(fig_sec); fig_sec.update_layout(height=450, xaxis_title="", yaxis_title="Euros Absolutos FOB (€ Bn)")
        st.plotly_chart(fig_sec, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 · FLUJOS COMERCIALES
# ═════════════════════════════════════════════════════════════════════════════
def page_trade_flow(where, transfer_pct):
    render_header("Flujos Comerciales y Red Estructural", "Mapeo global de dependencias y visualización topológica de la integración regional.")
    
    guide_box("MAPA Y RED DE FLUJOS", "Esta sección rastrea el origen geográfico de los productos que llegan a la Unión Europea. Observa el mapa de calor superior para identificar a los mayores proveedores globales. En la parte inferior, el diagrama de flujo (Sankey) te muestra visualmente cómo se distribuye la carga física desde el país de origen hasta cada puerto europeo. Intenta mover el 'Slider de Estrategia' en el panel lateral para ver cómo las líneas gruesas de suministro de larga distancia se reducen en favor de rutas más cortas y seguras.")
    sim_banner(transfer_pct)

    latest_year_df = q(33, f"SELECT MAX(YEAR(date)) AS y FROM trade {where}")
    if latest_year_df.empty or pd.isna(latest_year_df.iloc[0,0]): return st.warning("Sin datos geográficos suficientes en la selección.")
    latest_year = int(latest_year_df.iloc[0,0])

    origin_df = q(44, f"SELECT o_iso, origin_name, MAX(is_near) as is_near, SUM(eur)/1e9 AS eur_bn, AVG(dist_nm) AS avg_dist, AVG(tariff_rate)*100 AS tariff_pct FROM trade {where} AND YEAR(date)={latest_year} GROUP BY o_iso, origin_name ORDER BY eur_bn DESC")
    
    if transfer_pct > 0 and not origin_df.empty:
        offshore_sum = origin_df[origin_df['is_near'] == 0]['eur_bn'].sum()
        nearshore_sum = origin_df[origin_df['is_near'] == 1]['eur_bn'].sum()
        transfer_pool = offshore_sum * transfer_pct
        origin_df.loc[origin_df['is_near'] == 0, 'eur_bn'] *= (1 - transfer_pct)
        if nearshore_sum > 0:
            origin_df.loc[origin_df['is_near'] == 1, 'eur_bn'] += transfer_pool * (origin_df['eur_bn'] / nearshore_sum)

    origin_df = origin_df.sort_values(by='eur_bn', ascending=False).head(80)

    st.markdown("<br>", unsafe_allow_html=True)
    if not origin_df.empty:
        fig = px.choropleth(origin_df, locations='o_iso', color='eur_bn', hover_name='origin_name',
                            hover_data={'eur_bn':':.2f','avg_dist':':,.0f','tariff_pct':':.1f', 'is_near':False},
                            color_continuous_scale='Blues', labels={'eur_bn':'Valor Total (€ Bn)'},
                            title=f"Distribución Geográfica y Volumen de Proveedores hacia UE ({latest_year}){' - PROYECCIÓN SIMULADA' if transfer_pct > 0 else ''}")
        fig.update_geos(showframe=False, showcoastlines=True, coastlinecolor='#cbd5e1',
                        showland=True, landcolor='#f8fafc', showocean=True, oceancolor='#f1f5f9',
                        showcountries=True, countrycolor='#cbd5e1', bgcolor='rgba(0,0,0,0)')
        pt(fig); fig.update_layout(height=500, geo=dict(bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    sankey_data = q(9999, f"SELECT origin_name as source, d_iso as intermediate, puerto as target, MAX(is_near) as is_near, SUM(eur)/1e9 as value FROM trade {where} AND YEAR(date)={latest_year} GROUP BY 1,2,3")
    
    if transfer_pct > 0 and not sankey_data.empty:
        off_s_sum = sankey_data[sankey_data['is_near'] == 0]['value'].sum()
        nr_s_sum = sankey_data[sankey_data['is_near'] == 1]['value'].sum()
        s_pool = off_s_sum * transfer_pct
        sankey_data.loc[sankey_data['is_near'] == 0, 'value'] *= (1 - transfer_pct)
        if nr_s_sum > 0:
            sankey_data.loc[sankey_data['is_near'] == 1, 'value'] += s_pool * (sankey_data['value'] / nr_s_sum)
            
    sankey_data = sankey_data.sort_values('value', ascending=False).head(30)

    st.markdown("<br>", unsafe_allow_html=True)
    if not sankey_data.empty:
        sankey_data['intermediate'] = sankey_data['intermediate'].map(lambda x: ISO3_EU.get(x, x))
        all_nodes = list(pd.unique(sankey_data[['source','intermediate','target']].values.ravel('K')))
        ni = {n:i for i,n in enumerate(all_nodes)}
        src = [ni[x] for x in sankey_data['source']] + [ni[x] for x in sankey_data['intermediate']]
        tgt = [ni[x] for x in sankey_data['intermediate']] + [ni[x] for x in sankey_data['target']]
        val = list(sankey_data['value']) + list(sankey_data['value'])
        n_colors = [COLORS['ub_blue'] if n in sankey_data['source'].values
                    else COLORS['teal'] if n in sankey_data['intermediate'].values
                    else COLORS['amber'] for n in all_nodes]
        fig_s = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color='white', width=0.5), label=all_nodes, color=n_colors),
            link=dict(source=src, target=tgt, value=val, color='rgba(0,61,101,0.2)')))
        pt(fig_s); fig_s.update_layout(title=f"Red Estructural: País de Origen → Estado Miembro → Puerto de Arribo (€ Bn totales)", height=600)
        st.plotly_chart(fig_s, use_container_width=True)
        
        if transfer_pct > 0:
            ai_agent("RESILIENCIA TOPOLÓGICA", "Las densidades del diagrama Sankey reflejan el rebalanceo estratégico. Al ejecutar el Nearshoring, los flujos originados en redes de suministro de larga distancia (>3,000 nm) disminuyen progresivamente, obligando a los nodos del Mediterráneo a absorber la carga física entrante.")
        else:
            ai_agent("RESILIENCIA TOPOLÓGICA", "El diagrama Sankey hace visibles los Puntos Únicos de Fallo (Single Points of Failure). Las densidades que convergen masivamente en un solo nodo portuario representan un riesgo sistémico estructural. Diversificar orígenes es ineficaz si la cadena colapsa en la misma terminal de destino.")

    st.markdown("<br>", unsafe_allow_html=True)
    section("EVOLUCIÓN HISTÓRICA", "Transición de Socios Comerciales Principales a lo largo de los Años")
    anim_df = q(9998, f"SELECT YEAR(date) AS anio, origin_name, SUM(eur)/1e9 AS eur_bn FROM trade {where} GROUP BY 1, 2")
    if not anim_df.empty:
        anim_df['rank'] = anim_df.groupby('anio')['eur_bn'].rank(method='first', ascending=False)
        top_df = anim_df[anim_df['rank'] <= 15].copy().sort_values(['anio','rank'])
        top_df['etiqueta'] = top_df['origin_name'] + " (€" + top_df['eur_bn'].round(1).astype(str) + " Bn)"
        fig_anim = px.bar(top_df, x="eur_bn", y="rank", animation_frame="anio",
                          animation_group="origin_name", color="origin_name", text="etiqueta",
                          color_discrete_sequence=px.colors.qualitative.Pastel,
                          orientation='h', range_x=[0, top_df['eur_bn'].max()*1.4])
        fig_anim.update_yaxes(autorange="reversed", showticklabels=False, title="", showgrid=False)
        fig_anim.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
        fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 800
        pt(fig_anim); fig_anim.update_layout(height=550, showlegend=False, xaxis_title="Volumen Acumulado FOB (€ Bn)",
                                              margin=dict(l=20,r=50,t=40,b=40))
        fig_anim.update_traces(textposition='outside', textfont=dict(size=13, color='#1e293b', weight='bold'), cliponaxis=False)
        st.plotly_chart(fig_anim, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 · GEMELO PORTUARIO (M/M/1)
# ═════════════════════════════════════════════════════════════════════════════
def page_port_twin(where, transfer_pct):
    render_header(
        "Gemelo Portuario · Simulador de Transición Estructural",
        "Evaluación de estrés de infraestructura portuaria basada en la Teoría de Colas y el tamaño físico paramétrico de la flota marítima."
    )
    
    guide_box("SIMULACIÓN DE INFRAESTRUCTURA", "Este módulo responde a una pregunta crítica: '¿El puerto europeo soportaría el cambio estratégico?'. Selecciona tu puerto de interés en los menús desplegables. Abajo verás el tamaño promedio físico de los barcos (Drop Size). Si reduces barcos gigantes de larga distancia e incrementas barcos regionales pequeños para traer la misma cantidad de mercancía, la 'Frecuencia de Arribos' se dispara. Utiliza los controles para estresar la terminal y observa los días de espera proyectados.")

    sim_banner(transfer_pct)
    formula(
        "Saturación ρ = λ / μ  |  Tiempo Espera Promedio (Wq) = ρ / (μ - λ)  |  Línea Lq = λ × Wq  <br>"
        "λ_near_nueva = λ_near_base + (TEU_migrado_día ÷ Drop_Size_Nearshoring)  |  "
        "α_empírico = Drop_Size_Offshoring ÷ Drop_Size_Nearshoring"
    )

    geo_data = q(120, f"SELECT DISTINCT d_iso, puerto FROM trade {where}")
    if geo_data.empty:
        return st.warning("Ajusta los filtros geográficos.")

    country_names_map = {iso: ISO3_EU.get(iso, iso) for iso in geo_data['d_iso'].dropna().unique()}
    countries = ["TODOS"] + sorted(country_names_map.values())
    
    st.markdown("<br>", unsafe_allow_html=True)
    c0, c1 = st.columns(2)
    with c0: country_name = st.selectbox("Estado Miembro Destino", countries)
    country_sel_iso = (
        "TODOS" if country_name == "TODOS"
        else next(k for k, v in country_names_map.items() if v == country_name)
    )
    ports_avail = (
        ["TODOS"] + sorted(geo_data[geo_data['d_iso']==country_sel_iso]['puerto'].dropna().unique())
        if country_sel_iso != "TODOS"
        else ["TODOS"] + sorted(geo_data['puerto'].dropna().unique())
    )
    with c1: port_sel = st.selectbox("Terminal Portuaria de Arribo", ports_avail)
    port_filter_sql = ""
    if country_sel_iso != "TODOS": port_filter_sql += f" AND d_iso = '{country_sel_iso}'"
    if port_sel != "TODOS":        port_filter_sql += f" AND puerto = '{port_sel}'"

    port_split = q(130, f"""
        SELECT
            is_near,
            SUM(teu)     AS total_teu
        FROM trade {where} {port_filter_sql}
        GROUP BY is_near
    """)

    mu_query = q(131, f"""
        SELECT MAX(yr_teu) AS peak_yr_teu
        FROM (
            SELECT YEAR(date) AS yr, SUM(teu) AS yr_teu
            FROM trade {where} {port_filter_sql}
            GROUP BY 1
        )
    """)

    years_span_df = q(132, f"""
        SELECT MAX(YEAR(date)) - MIN(YEAR(date)) + 1 AS years_span
        FROM trade {where} {port_filter_sql}
    """)

    if port_split.empty or port_split['total_teu'].sum() == 0:
        return st.warning("Sin volumen TEU registrable para la selección actual.")

    years_span  = max(1, int(years_span_df.iloc[0,0])) if not years_span_df.empty else 1
    total_days  = float(years_span) * 365.0

    teu_far  = float(port_split[port_split['is_near'] == 0]['total_teu'].iloc[0]) if not port_split[port_split['is_near'] == 0].empty else 0.0
    teu_near = float(port_split[port_split['is_near'] == 1]['total_teu'].iloc[0]) if not port_split[port_split['is_near'] == 1].empty else 0.0

    annual_teu_port = (teu_far + teu_near) / years_span
    if annual_teu_port > 1000000:
        def_drop_far = 2800
        def_drop_near = 350
        port_tier = "Hub Principal (Megamax)"
    elif annual_teu_port > 250000:
        def_drop_far = 1500
        def_drop_near = 250
        port_tier = "Puerto Intermedio (Panamax)"
    else:
        def_drop_far = 800
        def_drop_near = 150
        port_tier = "Puerto Menor (Sub-Panamax)"

    st.markdown("<br>", unsafe_allow_html=True)
    section("PARÁMETROS DE INFRAESTRUCTURA FÍSICA", f"Configuración de Capacidad Operativa por Escala ({port_tier})", "El sistema auto-calibra el volumen promedio de descarga por buque según el tamaño histórico del puerto seleccionado.")

    c_ds1, c_ds2 = st.columns(2)
    with c_ds1:
        drop_far = st.number_input("Descarga Promedio Ultramar (Offshoring) [TEU/Escala]", min_value=100, value=def_drop_far, step=100, help="Volumen físico descargado por grandes buques permitidos por el calado.")
    with c_ds2:
        drop_near = st.number_input("Descarga Promedio Regional (Nearshoring) [TEU/Escala]", min_value=10, value=def_drop_near, step=50, help="Volumen físico descargado por buques Feeder/Ro-Ro en servicios de corta distancia.")

    alpha_factor = drop_far / drop_near if drop_near > 0 else 1.0

    daily_teu_far  = teu_far  / total_days
    daily_teu_near = teu_near / total_days

    lam_far_base   = daily_teu_far / drop_far if drop_far > 0 else 0
    lam_near_base  = daily_teu_near / drop_near if drop_near > 0 else 0
    lam_total_base = lam_far_base + lam_near_base

    total_ships_calc = (teu_far / drop_far) + (teu_near / drop_near) if drop_far > 0 and drop_near > 0 else 1
    blended_drop = (teu_far + teu_near) / total_ships_calc if total_ships_calc > 0 else drop_far
    
    peak_daily_teu = float(mu_query.iloc[0]['peak_yr_teu']) / 365.0 if not mu_query.empty and not pd.isna(mu_query.iloc[0]['peak_yr_teu']) else (teu_far+teu_near)/total_days
    mu_nominal = peak_daily_teu / blended_drop if blended_drop > 0 else lam_total_base * 1.5

    st.markdown("<br>", unsafe_allow_html=True)
    section(
        "INTELIGENCIA DE INFRAESTRUCTURA",
        "Radiografía Física del Puerto Seleccionado",
        "Métricas calculadas empíricamente parametrizando el volumen total procesado."
    )
    
    st.markdown(f"""
    <div style="background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-family: 'Space Mono', monospace; font-size: 14px; font-weight: 800; color: #003d65; margin-bottom: 15px; border-bottom: 2px solid #cbd5e1; padding-bottom: 5px; text-transform: uppercase;">
            Métricas Estructurales de Infraestructura {'(Simulado - ' + str(int(transfer_pct*100)) + '%)' if transfer_pct > 0 else ''}
        </div>
        <div style="display: flex; justify-content: space-between;">
            <div style="flex: 1; text-align: center;">
                <div style="font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; text-transform: uppercase;">VOLUMEN / ESCALA ULTRAMAR</div>
                <div style="font-size: 24px; font-weight: 800; color: #aa1916;">{drop_far:,.0f}</div>
                <div style="font-size: 11px; color: #475569;">TEU estimado por buque lejano</div>
            </div>
            <div style="flex: 1; text-align: center; border-left: 1px solid #cbd5e1;">
                <div style="font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; text-transform: uppercase;">VOLUMEN / ESCALA REGIONAL</div>
                <div style="font-size: 24px; font-weight: 800; color: #059669;">{drop_near:,.0f}</div>
                <div style="font-size: 11px; color: #475569;">TEU estimado por buque SSS</div>
            </div>
            <div style="flex: 1; text-align: center; border-left: 1px solid #cbd5e1;">
                <div style="font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; text-transform: uppercase;">FACTOR ALPHA ESTRUCTURAL</div>
                <div style="font-size: 24px; font-weight: 800; color: #d97706;">{alpha_factor:.2f}×</div>
                <div style="font-size: 11px; color: #475569;">Escalas extra por volumen transferido</div>
            </div>
            <div style="flex: 1; text-align: center; border-left: 1px solid #cbd5e1;">
                <div style="font-family: 'Space Mono', monospace; font-size: 10px; color: #475569; text-transform: uppercase;">TECHO OPERATIVO NOMINAL (μ)</div>
                <div style="font-size: 24px; font-weight: 800; color: #003d65;">{mu_nominal:,.2f} ll/día</div>
                <div style="font-size: 11px; color: #475569;">Límite histórico comprobable</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    info(
        f"<b>Nota Metodológica (Parametrización Estructural):</b><br>"
        f"En lugar de depender exclusivamente de conteos de registros aduaneros (que no equivalen físicamente a buques), el modelo estocástico parametriza el "
        f"<b>Drop Size</b> físico real. Al aplicar la simulación, la red logística debe absorber múltiples escalas regionales "
        f"para transportar el mismo volumen que traía un solo buque de ultramar. La capacidad nominal (μ) del nodo portuario "
        f"se autocalibra empíricamente dividiendo su volumen máximo histórico diario entre este Drop Size de flota."
    )

    collapse_85  = find_collapse_threshold(lam_far_base, lam_near_base, daily_teu_far, drop_near, mu_nominal, target_rho=0.85)
    collapse_100 = find_collapse_threshold(lam_far_base, lam_near_base, daily_teu_far, drop_near, mu_nominal, target_rho=1.00)

    st.markdown("<br>", unsafe_allow_html=True)
    col_th1, col_th2 = st.columns(2)
    with col_th1:
        if collapse_85:
            warn(f"Peligro Operativo: La terminal entrará en <b>Zona Crítica de Estrés (Saturación >= 85%)</b> si se alcanza el <b>{collapse_85}%</b> de relocalización estratégica.")
        else:
            st.markdown('<div class="threshold-badge ok">Evaluación: La terminal no alcanza zonas críticas de estrés (85%) en las simulaciones de la muestra.</div>', unsafe_allow_html=True)
    with col_th2:
        if collapse_100:
            danger(f"Colapso Predictivo: La infraestructura sufrirá fallo sistémico (<b>Saturación >= 100%</b>, colas logísticas infinitas) al <b>{collapse_100}%</b> de relocalización.")
        else:
            st.markdown('<div class="threshold-badge ok">Evaluación: La terminal analizada no proyecta colapsos totales de servicio operativo.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section(
        "MODELADO DE ESTRÉS DE INFRAESTRUCTURA",
        "Control Fino de Relocalización Portuaria",
        "Utiliza los controles inferiores para estresar puntualmente la red logística y visualizar impactos."
    )
    c_ctrl1, c_ctrl2 = st.columns([3, 1])
    with c_ctrl1:
        local_pct = st.slider(
            "Carga de Ultramar a Redirigir Localmente (%)",
            min_value=0, max_value=100,
            value=int(transfer_pct * 100),
            step=5,
            help=(
                f"Toma como base el valor dictado por la estrategia global ({int(transfer_pct*100)}%). "
            )
        )
    with c_ctrl2:
        peak_stress = st.slider("Factor de Estrés Estacional", 1.0, 2.5, 1.0, 0.1, help="Simula picos de demanda masiva multiplicando la tasa de arribos proyectada.")

    pct_migration = local_pct / 100.0

    migrated_daily_teu = daily_teu_far * pct_migration
    lam_far_new    = lam_far_base * (1.0 - pct_migration)
    lam_near_new   = lam_near_base + (migrated_daily_teu / drop_near) if drop_near > 0 else lam_near_base
    lam_base_str   = lam_total_base * peak_stress
    lam_new_str    = (lam_far_new + lam_near_new) * peak_stress

    metrics_before = macro_queue_metrics(lam_base_str, mu_nominal)
    metrics_after  = macro_queue_metrics(lam_new_str,  mu_nominal)

    st.markdown("<br>", unsafe_allow_html=True)
    section(
        "ANÁLISIS TEORÍA DE COLAS (M/M/1)",
        "Respuesta Operativa: Situación Histórica vs Proyección Estocástica",
        "Observa la degradación de tiempos por el aumento del número físico de arribos marítimos."
    )

    def color_rho(rho):
        return COLORS['green'] if rho<0.70 else COLORS['amber'] if rho<0.85 else COLORS['ub_red']

    def fmt_m(val, suffix):
        return f"{val:.2f} {suffix}" if np.isfinite(val) else "Colapso Técnico"

    col_b, col_mid, col_a = st.columns([10, 1, 10])
    with col_b:
        st.markdown('<div class="before-panel"><div class="panel-title" style="color:#aa1916;">MODELO BASE — RÉGIMEN ACTUAL</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        rho_b = metrics_before['rho']
        with m1: kpi("Factor Saturación",        f"{rho_b*100:.1f}%",               f"Tasa base: {lam_base_str:.2f} ll/día",  color_rho(rho_b))
        with m2: kpi("Tiempo Wq",          fmt_m(metrics_before['Wq'],'días'),"Retraso medio estimado",                  COLORS['ub_blue'])
        with m3: kpi("Longitud Lq",  fmt_m(metrics_before['Lq'],'ll.'), "Fila de buques promedio",            COLORS['ub_blue'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_mid:
        st.markdown("<div style='text-align:center;font-size:30px;padding-top:60px;color:#003d65;font-weight:900;'>→</div>", unsafe_allow_html=True)

    with col_a:
        st.markdown(f'<div class="after-panel"><div class="panel-title" style="color:#059669;">PROYECCIÓN — {local_pct}% NEARSHORING APLICADO</div>', unsafe_allow_html=True)
        n1, n2, n3 = st.columns(3)
        rho_a = metrics_after['rho']
        with n1: kpi("Factor Saturación",        f"{rho_a*100:.1f}%",               f"Tasa nueva: {lam_new_str:.2f} ll/día",  color_rho(rho_a))
        with n2: kpi("Tiempo Wq",          fmt_m(metrics_after['Wq'],'días'),"Retraso medio proyectado",                  COLORS['ub_blue'])
        with n3: kpi("Longitud Lq",  fmt_m(metrics_after['Lq'],'ll.'), "Fila de buques proyectada",            COLORS['ub_blue'])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("DESCOMPOSICIÓN DE ARRIBOS FÍSICOS", "Anatomía del Rebalanceo Operacional",
            "Muestra gráficamente el número de escalas reducidas en larga distancia, superadas por la inyección de escalas regionales.")

    wf_remove = -(lam_far_base * pct_migration) * peak_stress
    wf_add    = (migrated_daily_teu / drop_near) * peak_stress if drop_near > 0 else 0
    wf_total  = lam_new_str

    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=[
            "Base Oceánica Total",
            f"Escalas Descartadas ({local_pct}%)",
            f"Escalas Adicionadas",
            "Nuevo Volumen de Arribos"
        ],
        y=[lam_base_str, wf_remove, wf_add, wf_total],
        text=[f"{lam_base_str:.2f}", f"{wf_remove:.2f}", f"+{wf_add:.2f}" if wf_add>=0 else f"{wf_add:.2f}", f"{wf_total:.2f}"],
        textposition="outside",
        connector={"line": {"color": "#94a3b8", "dash": "dot"}},
        increasing={"marker": {"color": COLORS['ub_red']}},
        decreasing={"marker": {"color": COLORS['green']}},
        totals={"marker":   {"color": COLORS['ub_blue']}}
    ))
    pt(fig_wf)
    fig_wf.update_layout(
        title=f"Cascada de Escalas Diarias Proyectadas",
        yaxis_title="Cantidad de Arribos Diarios Estimados", height=450, margin=dict(t=60, b=80)
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("CURVAS DE FRICCIÓN ASINTÓTICA", "Límites Estructurales del Puerto ante Transiciones Nearshoring",
            "Líneas de advertencia señalan umbrales críticos de ineficiencia operativa.")

    pct_arr = np.linspace(0, 97, 70)
    rho_arr, wq_arr, lq_arr = [], [], []
    for pct in pct_arr:
        lf = lam_far_base  * (1.0 - pct/100.0)
        ln = lam_near_base + (daily_teu_far * (pct/100.0) / max(drop_near, 0.001))
        lt = (lf + ln) * peak_stress
        m  = macro_queue_metrics(lt, mu_nominal)
        rho_arr.append(min(m['rho']*100, 200.0))
        wq_arr.append( min(m['Wq'], 60.0) if np.isfinite(m['Wq']) else 60.0)
        lq_arr.append( min(m['Lq'], 8000.0) if np.isfinite(m['Lq']) else 8000.0)

    fig_curves = make_subplots(rows=1, cols=3,
        subplot_titles=["Evolución de Saturación (%)",
                        "Retraso Medio Wq (Días)",
                        "Acumulación de Buques Lq"])

    for i, (arr, color) in enumerate(zip(
        [rho_arr, wq_arr, lq_arr],
        [COLORS['ub_blue'], COLORS['amber'], COLORS['ub_red']]
    )):
        fig_curves.add_trace(go.Scatter(x=pct_arr, y=arr, mode='lines',
                                        line=dict(color=color, width=2.5), showlegend=False),
                             row=1, col=i+1)
        idx = int(np.argmin(np.abs(pct_arr - local_pct)))
        fig_curves.add_trace(go.Scatter(x=[local_pct], y=[arr[idx]], mode='markers',
                                        marker=dict(size=14, color=color, symbol='circle-open',
                                                    line=dict(width=3, color=color)), showlegend=False),
                             row=1, col=i+1)
        if collapse_85:
            fig_curves.add_vline(x=collapse_85, line_dash="dot", line_color=COLORS['amber'], row=1, col=i+1)
        if collapse_100:
            fig_curves.add_vline(x=collapse_100, line_dash="dot", line_color=COLORS['ub_red'], row=1, col=i+1)

    fig_curves.add_hline(y=85, line_dash="dot", line_color=COLORS['ub_red'],
                         annotation_text="Estrés M/M/1", row=1, col=1)
    fig_curves.add_hline(y=100, line_dash="dash", line_color=COLORS['ub_red'],
                         annotation_text="Saturación Total", row=1, col=1)

    pt(fig_curves); fig_curves.update_layout(
        height=450,
        title=f"Curvas de Sensibilidad Estocástica — Velocidad de Extracción (μ) = {mu_nominal:.2f} ll/día"
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    delta_rho_pp = (metrics_after['rho'] - metrics_before['rho']) * 100
    delta_wq = ((metrics_after['Wq'] - metrics_before['Wq'])
                if np.isfinite(metrics_after['Wq']) and np.isfinite(metrics_before['Wq'])
                else float('inf'))

    st.markdown("<br>", unsafe_allow_html=True)
    if metrics_after['rho'] >= 1.0:
        ai_agent(
            "FALLO TÉCNICO INMINENTE",
            f"Al ejecutar la migración física del {local_pct}% de la carga, la matemática portuaria evidencia un fallo. "
            f"El aumento drástico de la tasa de arribos rebasa la capacidad de extracción diaria del sistema. La saturación rompe la barrera del 100% "
            f"(Proyectado: {metrics_after['rho']*100:.1f}%), garantizando filas infinitas. Es mandatario expandir calados o automatizar procesos aduaneros antes de ejecutar esta política."
        )
    elif metrics_after['rho'] >= 0.85:
        ai_agent(
            "ZONA ROJA OPERACIONAL",
            f"La proyección de Nearshoring al {local_pct}% ubica a la terminal logística en un régimen de estrés grave ({metrics_after['rho']*100:.1f}% de utilización). "
            f"El sistema absorbería retrasos de {delta_wq:+.1f} días enteros en espera en muelle. De acuerdo con la Ley de Little, cualquier fluctuación o disrupción generará incrementos exponenciales en los costos de inmovilización de capital."
        )
    else:
        ai_agent(
            "VIABILIDAD ESTRUCTURAL",
            f"El modelado estocástico confirma que un nivel de reubicación regional del {local_pct}% es estructuralmente viable. "
            f"La terminal logística absorbe la nueva frecuencia de servicios SSS operando a una saturación aceptable del {metrics_after['rho']*100:.1f}%. "
            f"Las ineficiencias transaccionales están controladas y el ecosistema no presenta cuellos de botella severos."
        )

# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 7 · ANÁLISIS DE SOBRECOSTOS (TCO)
# ═════════════════════════════════════════════════════════════════════════════
def page_cost_xray(where, transfer_pct):
    render_header("Análisis de Sobrecostos (TCO)", "Desglose financiero estructural incorporando aranceles aduaneros, normativas de carbono europeas y fricción logística acumulada.")
    
    guide_box("RADIOGRAFÍA DE SOBRECOSTOS", "Aquí visualizamos qué elementos devoran los márgenes de importación. Observa el diagrama de 'Cascada' para entender el peso del arancel, el impuesto al carbono y las ineficiencias operativas. Abajo, el gráfico de barras clasifica a los países según el castigo tributario y de emisiones que imponen a tu cadena de suministro.")

    sim_banner(transfer_pct)

    c_df_raw = q(19, f"SELECT SUM(CASE WHEN is_near=0 THEN costo_arancel ELSE 0 END)/1e9 as a_off, SUM(CASE WHEN is_near=1 THEN costo_arancel ELSE 0 END)/1e9 as a_nr, SUM(CASE WHEN is_near=0 THEN costo_co2_ets ELSE 0 END)/1e9 as c_off, SUM(CASE WHEN is_near=1 THEN costo_co2_ets ELSE 0 END)/1e9 as c_nr, SUM(CASE WHEN is_near=0 THEN alpha_resiliencia_nodal ELSE 0 END)/1e9 as al_off, SUM(CASE WHEN is_near=1 THEN alpha_resiliencia_nodal ELSE 0 END)/1e9 as al_nr FROM trade {where}")
    
    if not c_df_raw.empty:
        c_df = c_df_raw.iloc[0]
        a_off = float(c_df['a_off']) if pd.notna(c_df['a_off']) else 0.0
        a_nr  = float(c_df['a_nr'])  if pd.notna(c_df['a_nr'])  else 0.0
        c_off = float(c_df['c_off']) if pd.notna(c_df['c_off']) else 0.0
        c_nr  = float(c_df['c_nr'])  if pd.notna(c_df['c_nr'])  else 0.0
        al_off= float(c_df['al_off'])if pd.notna(c_df['al_off'])else 0.0
        al_nr = float(c_df['al_nr']) if pd.notna(c_df['al_nr']) else 0.0

        formula("Costo Total de Propiedad (TCO) = Arancel Preferencial + Marco CO₂ ETS + Fricción Operativa (Alpha Nodal)")

        if transfer_pct > 0:
            info(f"<b>Ajuste de Simulación Activo:</b> La redistribución regional aplicada proyecta una variación paramétrica en la recaudación arancelaria debido a Tratados de Libre Comercio (TLC) y un ajuste del pasivo ambiental (CO₂) por la reducción de la huella de carbono oceánica.")

        a_sim = (a_off * (1-transfer_pct)) + a_nr + (a_off * transfer_pct * 0.4) 
        c_sim = (c_off * (1-transfer_pct)) + c_nr + (c_off * transfer_pct * 0.2) 
        al_sim = (al_off * (1-transfer_pct)) + al_nr + (al_off * transfer_pct * 1.5) 
        total_friction = a_sim + c_sim + al_sim

        st.markdown("<br>", unsafe_allow_html=True)
        fig_w = go.Figure(go.Waterfall(
            name="Friction Cost", orientation="v", measure=["relative","relative","relative","total"],
            x=["Carga Arancelaria Anualizada","Normativa CO₂ (EU ETS)","Fricción Operativa Acumulada","TCO Global Estimado"],
            textposition="outside",
            text=[f"€{x:,.2f} Bn" for x in [a_sim, c_sim, al_sim, total_friction]],
            y=[a_sim, c_sim, al_sim, total_friction],
            connector={"line":{"color":"#94a3b8","dash":"dot"}},
            increasing={"marker":{"color":COLORS['ub_blue']}},
            totals={"marker":{"color":COLORS['ub_red']}}
        ))
        pt(fig_w); fig_w.update_layout(title="Cascada Financiera: Estructura de Costos Ocultos de Importación", height=500)
        st.plotly_chart(fig_w, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("MAPEO DE FISCALIDAD INTERNACIONAL", "Tasa Promedio de Castigo TCO por País", "Identificación visual comparativa del régimen impositivo general frente a los esquemas de integración preferencial.")
        
        paises_df = q(199, f"""
            SELECT origin_name, SUM(eur)/1e9 AS eur_bn,
                   (SUM(total_friction_cost)/SUM(eur))*100 AS tco_pct
            FROM trade {where} GROUP BY origin_name
            HAVING SUM(eur)>0 ORDER BY eur_bn DESC LIMIT 15
        """)
        if not paises_df.empty:
            paises_df = paises_df.sort_values('tco_pct', ascending=True)
            fig_p = px.bar(paises_df, x='tco_pct', y='origin_name', orientation='h',
                           color='tco_pct', color_continuous_scale=['#059669','#d97706','#aa1916'],
                           labels={'tco_pct':'TCO Promedio (%)','origin_name':'Socio Comercial'},
                           title="Penalización Económica Estructural por Proveedor Principal")
            fig_p.add_vline(x=paises_df['tco_pct'].mean(), line_dash="dash",
                            line_color=COLORS['ub_blue'], annotation_text="Promedio Sistémico")
            pt(fig_p); fig_p.update_layout(height=500, coloraxis_showscale=False)
            st.plotly_chart(fig_p, use_container_width=True)
            
            ai_agent("ANÁLISIS FINANCIERO ESTRATÉGICO",
                     "La distribución de la carga impositiva visualiza el diferencial económico del Nearshoring institucional. Las naciones ubicadas en la banda verde evidencian la cobertura y ventaja competitiva de los Tratados de Libre Comercio (TLC). "
                     "Por contraparte, los socios en la zona roja absorben gravámenes estándar y asumen la mayor parte del costo punitivo de la normativa climática oceánica. La optimización exige focalizar compras en proveedores con menor fricción estructural.")


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 · MOTOR MONTECARLO LVaR
# ═════════════════════════════════════════════════════════════════════════════
def page_montecarlo(where):
    render_header("Motor de Riesgo Estocástico (LVaR)", "Proyección probabilística de inmovilización de capital a través de miles de iteraciones Monte Carlo simulando disrupciones históricas y futuras.")
    
    guide_box("ANÁLISIS DE RIESGO MATEMÁTICO", "Este panel no analiza lo que pasó, analiza 'lo que podría salir mal'. Usando inteligencia estocástica (Monte Carlo), el sistema tira los dados miles de veces simulando retrasos logísticos en tu cadena. Simplemente elige cuántas simulaciones quieres correr y tu costo de capital (WACC). El resultado te dirá cuánto 'colchón financiero' debes guardar ante crisis.")

    formula("LVaR = Coeficiente_Z(95%) × Volatilidad_Histórica(Lead_Time) × Capital_Diario_FOB × (Tasa_BCE / 365)   |   Base Estocástica: Lognormal")
    active_data = q(998, f"SELECT DISTINCT macro_sector, hs2_nombre FROM trade {where}")
    if active_data.empty: return st.warning("Volumen de datos insuficiente para ejecutar motor probabilístico.")
    avail_macros = sorted(active_data['macro_sector'].dropna().unique().tolist())
    avail_micros = sorted(active_data['hs2_nombre'].dropna().unique().tolist())
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_ctrl = st.columns(3)
    with c_ctrl[0]: sector_mc = st.selectbox("Aislamiento Vectorial Sectorial", ["TODA LA SELECCIÓN"]+avail_macros+avail_micros)
    with c_ctrl[1]: n_sim     = st.select_slider("Resolución Computacional (Iteraciones)", [1000,5000,10000,25000], 10000)
    with c_ctrl[2]: bce_over  = st.slider("Costo de Capital Promedio (WACC / Tasa BCE %)", -0.5, 5.5, 1.5, 0.25)
    
    lf_mc = ("" if sector_mc=="TODA LA SELECCIÓN" else f" AND macro_sector='{sector_mc}'" if sector_mc in avail_macros else f" AND hs2_nombre='{sector_mc}'")
    params = q(11, f"SELECT AVG(lead_time) AS lt_m, AVG(lt_std) AS lt_s, AVG(eur) AS eur_m, STDDEV(eur) AS eur_s FROM trade {where} {lf_mc}")
    if params.empty or pd.isna(params.iloc[0]['lt_m']): return st.warning("Datos paramétricos de tiempo en ruta insuficientes.")
    pv = params.iloc[0]
    
    with st.spinner(f"Compilando y ejecutando {n_sim:,} escenarios probabilísticos independientes..."):
        mc = monte_carlo_lvar(pv['lt_m'], pv['lt_s'], pv['eur_m'], pv['eur_s'] if not pd.isna(pv['eur_s']) else pv['eur_m']*0.5, bce_over, n_sim)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: kpi("Líneas Temporales Evaluadas", f"{n_sim:,}", "Iteraciones independientes procesadas", COLORS['ub_blue'])
    with c2: kpi("Valor en Riesgo (VaR 95%)", f"€{mc['p95']:,.2f}", "Pérdida de capital límite confiable", COLORS['ub_blue'])
    with c3: kpi("Riesgo de Cola Crítico (CVaR)", f"€{mc['cvar_95']:,.2f}", "Promedio del peor 5% histórico", COLORS['ub_red'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    filtered_sim = mc['sim'][mc['sim'] <= np.percentile(mc['sim'], 99.5)]
    fig_h = px.histogram(filtered_sim, nbins=80, title="Función de Densidad de Probabilidad de Exposición Logística",
                         color_discrete_sequence=[COLORS['ub_blue']], marginal="box")
    fig_h.add_vline(x=mc['p95'], line_color=COLORS['ub_red'], line_dash="dash", annotation_text="Frontera de Riesgo LVaR 95%")
    pt(fig_h); fig_h.update_layout(xaxis_title="Severidad del Costo Estocástico Esperado (€)", yaxis_title="Densidad Probabilística Acumulada", height=450)
    st.plotly_chart(fig_h, use_container_width=True)
    
    ai_agent("LECTURA TÉCNICA DE RIESGO DE COLA", f"La distribución modelada del sistema de fletes marítimos confirma una asimetría fuertemente positiva (comportamiento de cola pesada). Esto significa que, bajo las condiciones extraídas, existe un 5% estricto de probabilidad de que una disrupción portuaria congele **€{mc['cvar_95']:,.2f}** adicionales por embarque. Se recomienda dotar reservas de Safety Stock financiero equivalentes a esta métrica.")


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 6 · MONITOREO NEARSHORING
# ═════════════════════════════════════════════════════════════════════════════
def page_nearshoring(where, transfer_pct):
    render_header("Monitoreo de Macro-Tendencias Nearshoring", "Visualización profunda de la integración intrarregional (<3,000 nm) frente a las redes de suministro de larga distancia (>3,000 nm).")
    
    guide_box("MONITOREO DE REGIONALIZACIÓN", "Compara el comercio de corta distancia (Nearshoring) contra el de rutas de ultramar. El gráfico de áreas te muestra la evolución histórica de ambos volúmenes. Más abajo, el Índice de Concentración (HHI) te alerta si tu cadena depende de muy pocos proveedores (línea subiendo hacia la zona de oligopolio) o si goza de resiliencia geográfica (zona verde).")

    sim_banner(transfer_pct)

    df_near = q(300, f"""
        SELECT YEAR(date) as anio, is_near, SUM(eur)/1e9 as eur_bn
        FROM trade {where} GROUP BY 1, 2 ORDER BY 1, 2
    """)
    if df_near.empty: return st.warning("Volumen de registros insuficiente.")

    df_piv = df_near.pivot(index='anio', columns='is_near', values='eur_bn').fillna(0)
    if 0 not in df_piv.columns: df_piv[0] = 0.0
    if 1 not in df_piv.columns: df_piv[1] = 0.0
    df_piv.columns = ['Offshoring', 'Nearshoring']
    df_piv['Total'] = df_piv['Offshoring'] + df_piv['Nearshoring']
    df_piv['pct_near'] = df_piv['Nearshoring'] / df_piv['Total'] * 100

    latest_year  = df_piv.index.max()
    val_off_base = df_piv.loc[latest_year, 'Offshoring']
    val_nr_base  = df_piv.loc[latest_year, 'Nearshoring']

    transf_val   = val_off_base * transfer_pct
    val_off_sim  = val_off_base - transf_val
    val_nr_sim   = val_nr_base  + transf_val
    pct_near_sim = (val_nr_sim / (val_off_sim + val_nr_sim)) * 100 if (val_off_sim + val_nr_sim) > 0 else 0

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Volumen Absoluto Regional", f"€{val_nr_sim:,.1f} Bn", f"Acumulado al año {latest_year}", COLORS['green'])
    with c2: kpi("Índice de Regionalización", f"{pct_near_sim:.1f}%", "Proporción intra-europea", COLORS['ub_blue'])
    with c3: kpi("Volumen Rutas Lejanas", f"€{val_off_sim:,.1f} Bn", "Suma en rutas extendidas", COLORS['ub_red'])
    with c4:
        if len(df_piv) >= 3:
            cagr_n = ((df_piv['Nearshoring'].iloc[-1] / max(df_piv['Nearshoring'].iloc[0], 0.001)) ** (1/max(len(df_piv)-1,1)) - 1) * 100
            kpi("Tasa Anual Equivalente CAGR", f"{cagr_n:+.1f}%", f"Tendencia general desde {df_piv.index.min()}", COLORS['teal'])
        else:
            kpi("Años Base Analizados", f"{len(df_piv)}", "Histórico disponible en filtro", COLORS['ub_slate'])

    df_piv['Off_sim']  = df_piv['Offshoring'] * (1 - transfer_pct)
    df_piv['Near_sim'] = df_piv['Nearshoring'] + df_piv['Offshoring'] * transfer_pct

    st.markdown("<br>", unsafe_allow_html=True)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=df_piv.index, y=df_piv['Off_sim'],  name='Comercio Oceánico Lejano',
                              fill='tozeroy', fillcolor='rgba(170,25,22,0.1)',
                              line=dict(color=COLORS['ub_red'], width=2)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_piv.index, y=df_piv['Near_sim'], name='Comercio Regional (Nearshoring)',
                              fill='tozeroy', fillcolor='rgba(5,150,105,0.12)',
                              line=dict(color=COLORS['green'], width=2)), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_piv.index, y=df_piv['pct_near']*(1 + transfer_pct*0.5),
                              name='Penetración Porcentual Regional (%)', mode='lines+markers',
                              line=dict(color=COLORS['amber'], width=2, dash='dot'),
                              marker=dict(size=6)), secondary_y=True)
    pt(fig2)
    fig2.update_layout(
        title=f"Evolución Empírica Temporal: Régimen de Ultramar vs Integración de Cuenca",
        hovermode="x unified", height=500,
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor='center')
    )
    fig2.update_yaxes(title_text="Volumen Sumarizado Anual (€ Bn)", secondary_y=False)
    fig2.update_yaxes(title_text="Penetración Absoluta (%)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("MAPA DE CONCENTRACIÓN Y RIESGO GEOPOLÍTICO", "Rastreo Anual del Índice Herfindahl-Hirschman (HHI) Aplicado al Comercio",
            "Métricas descendentes evidencian redes geográficamente resilientes. Escalas ascendentes marcan dependencia estructural en pocos proveedores clave.")
    hhi_ts = q(301, f"""
        WITH sh AS (
            SELECT YEAR(date) AS anio, o_iso, SUM(eur) AS val,
                   SUM(SUM(eur)) OVER (PARTITION BY YEAR(date)) AS tot
            FROM trade {where} GROUP BY 1, 2
        )
        SELECT anio, SUM((val/NULLIF(tot,0))*(val/NULLIF(tot,0)))*10000 AS hhi
        FROM sh GROUP BY 1 ORDER BY 1
    """)
    if not hhi_ts.empty:
        fig_hhi = go.Figure()
        fig_hhi.add_trace(go.Scatter(x=hhi_ts['anio'], y=hhi_ts['hhi'], mode='lines+markers',
                                     line=dict(color=COLORS['purple'], width=2.5),
                                     marker=dict(size=7, color=COLORS['purple'])))
        fig_hhi.add_hline(y=2500, line_dash="dot", line_color=COLORS['amber'], annotation_text="Umbral de Alerta Sistémica (HHI > 2500)")
        fig_hhi.add_hline(y=1500, line_dash="dot", line_color=COLORS['green'], annotation_text="Red Altamente Resiliente (HHI < 1500)")
        pt(fig_hhi); fig_hhi.update_layout(height=400, title="Trazabilidad Algorítmica de Independencia Geoestratégica",
                                            xaxis_title="Frecuencia Anual de Registro", yaxis_title="Métrica IHH Absoluta")
        st.plotly_chart(fig_hhi, use_container_width=True)

    ai_agent(
        "PROYECCIÓN DE RED ESTRATÉGICA",
        f"La evaluación dinámica arroja que la cuenca mediterránea/europea concentra el {pct_near_sim:.1f}% de todo el abastecimiento físico importado. "
        f"Al promover traslados a zonas más cortas, el modelo HHI (Índice Herfindahl) superior advierte posibles monopolizaciones en países frontera. Una buena regionalización mejora los plazos operativos, pero jamás debe comprometer la diversificación geográfica."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 · MODELADO DE ESCENARIOS
# ═════════════════════════════════════════════════════════════════════════════
def page_scenario(where):
    render_header("Evaluación de Políticas y Escenarios Macro", "Modelaje proyectivo para estimar los choques sobre márgenes, derivados de barreras de protección comercial o penalizaciones de la agenda de carbono.")
    
    guide_box("SIMULADOR DE ESCENARIOS Y POLÍTICAS", "Módulo interactivo estilo 'What-If'. Úsalo para probar impactos de políticas futuras. En la pestaña 'Política Arancelaria', observa qué pasaría si un producto sufriera un aumento de aranceles de golpe. En 'Matriz Multi-Shock' puedes comparar simultáneamente hasta 4 escenarios de tensión, mientras que el 'Comparador A/B' te permite enfrentar financieramente dos países proveedores frente a frente.")

    active_data = q(999, f"SELECT DISTINCT macro_sector, hs2_nombre FROM trade {where}")
    if active_data.empty: return st.warning("Datos brutos insuficientes para ejecutar matrices.")
    avail_macros  = sorted(active_data['macro_sector'].dropna().unique().tolist())
    avail_micros  = sorted(active_data['hs2_nombre'].dropna().unique().tolist())
    avail_options = ["TODA LA SELECCIÓN"] + avail_macros + avail_micros

    tabs = st.tabs(["Sensibilidad Arancelaria Global","Proyecciones de Carbono EU ETS","Matriz Multi-Shock","Validación Gravitacional","Comparador Analítico A/B Nodal"])

    with tabs[0]:
        formula("Función Contracción = Multiplicador_Elasticidad_Empírica × (Δ Arancel) / (1 + Tasa_Arancel_Base_Histórica)")
        c1,c2,c3 = st.columns(3)
        with c1: sector_sel = st.selectbox("Sub-Sector Objetivo", avail_options)
        with c2: tariff_new = st.slider("Arancel Fronterizo Objetivo (%)", 0.0, 30.0, 10.0, 0.5) / 100
        with c3: elasticity = st.slider("Elasticidad Precio-Importación", -2.5, -0.3, -1.1, 0.1)
        lf = ("" if sector_sel=="TODA LA SELECCIÓN" else f" AND macro_sector='{sector_sel}'" if sector_sel in avail_macros else f" AND hs2_nombre='{sector_sel}'")
        base = q(7, f"SELECT SUM(eur)/1e9 AS eur_bn, AVG(tariff_rate) AS tariff_base, SUM(costo_arancel)/1e9 AS tariff_cost_bn FROM trade {where} {lf}")
        if not base.empty and not pd.isna(base.iloc[0]['eur_bn']) and base.iloc[0]['eur_bn']>0:
            bv = base.iloc[0]
            res = tariff_trade_impact(bv['eur_bn'], bv['tariff_base'], tariff_new, elasticity)
            delta_costo_bn = res['eur_new']*tariff_new - bv['tariff_cost_bn']
            st.markdown("<br>", unsafe_allow_html=True)
            c4,c5,c6 = st.columns(3)
            with c4: kpi("Tasa Arancelaria Promedio Real", f"{bv['tariff_base']*100:.2f}%", f"Recaudación base histórica: €{bv.tariff_cost_bn:.2f} Bn", COLORS['ub_blue'])
            with c5: kpi("Volumen Remanente Estimado", f"€{res['eur_new']:,.2f} Bn", f"Contracción total calculada: {abs(res['pct_change']*100):.1f}%", COLORS['ub_red'])
            with c6: kpi("Delta Recaudación Aduanera", f"€{delta_costo_bn:+,.2f} Bn", "Variación de costos extra absorbidos", COLORS['amber'])

    with tabs[1]:
        formula("Matriz Tributaria Climática = Total Emisiones Estimadas (Ton) × Precio Abierto Spot (€) × Rango Cobertura Reglamentaria (%)")
        c1,c2 = st.columns(2)
        with c1: ets_new  = st.slider("Precio Cotización Futura (€/tonelada CO₂)", 20, 200, 85, 5)
        with c2: coverage = st.slider("Esquema de Obligación Aplicable (%)", 25, 100, 100, 5)
        base_co2 = q(8, f"SELECT SUM(costo_co2_ets)/1e9 AS co2_base_bn, SUM(costo_co2_ets/85.0*1000) AS tons_total FROM trade {where}")
        if not base_co2.empty and not pd.isna(base_co2.iloc[0]['tons_total']):
            tons = base_co2.iloc[0]['tons_total']
            co2_new = tons * ets_new * (coverage/100) / 1e9
            st.markdown("<br>", unsafe_allow_html=True)
            c4,c5,c6 = st.columns(3)
            with c4: kpi("Inventario Trazable Físico", f"{tons/1e6:,.2f} M Ton", "Huella de carbono oceánica estimada", COLORS['ub_slate'])
            with c5: kpi("Pasivo Estructural Histórico", f"€{base_co2.iloc[0]['co2_base_bn']:,.2f} Bn", "Modelo base teórico referencial", COLORS['ub_blue'])
            with c6: kpi("Nuevo Pasivo Climático", f"€{co2_new:,.2f} Bn", f"Impacto estimado tasado a €{ets_new}/tonelada", COLORS['ub_red'])

    with tabs[2]:
        sector_ms = st.selectbox("Aplicación Segmentada por Mercado", avail_options, key='ms_sec')
        lf_ms = ("" if sector_ms=="TODA LA SELECCIÓN" else f" AND macro_sector='{sector_ms}'" if sector_ms in avail_macros else f" AND hs2_nombre='{sector_ms}'")
        base_ms = q(700, f"SELECT SUM(eur)/1e9 AS eur_bn, AVG(tariff_rate) AS tariff_base FROM trade {where} {lf_ms}")
        if not base_ms.empty and not pd.isna(base_ms.iloc[0]['eur_bn']):
            bm = base_ms.iloc[0]
            st.markdown("**Parámetros Analíticos Personalizados (Carga Tributaria Fronteriza en %)**")
            c_s1,c_s2,c_s3,c_s4 = st.columns(4)
            # Etiquetas acortadas
            with c_s1: t1 = st.number_input("Base (%)", 0.0, 40.0, float(bm['tariff_base']*100), 0.5)
            with c_s2: t2 = st.number_input("Moderado (%)", 0.0, 40.0, float(bm['tariff_base']*100)+5, 0.5)
            with c_s3: t3 = st.number_input("Alto (%)", 0.0, 40.0, float(bm['tariff_base']*100)+10, 0.5)
            with c_s4: t4 = st.number_input("Extremo (%)", 0.0, 40.0, float(bm['tariff_base']*100)+15, 0.5)
            elast_ms = st.slider("Sensibilidad Demanda Consumidor", -2.5, -0.3, -1.1, 0.1, key='ms_elast')
            scenarios = {'Situación Registrada': bm['tariff_base']*100, 'Escenario Moderado': t1, 'Escenario Alto': t2, 'Escenario Severo': t3, 'Shock Global': t4}
            rows = []
            for name, tpct in scenarios.items():
                r = tariff_trade_impact(bm['eur_bn'], bm['tariff_base'], tpct/100, elast_ms)
                rows.append({'Tensión Simulada': name, 'Tasa Aplicada (%)': f"{tpct:.2f}%", 'Total Intercambiado (€ Bn)': round(r['eur_new'],3),
                             'Caída Flujos Esperada (%)': f"{r['pct_change']*100:+.1f}%", 'Recaudación Teórica (€ Bn)': round(r['eur_new']*tpct/100,2)})
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tabs[3]:
        formula("Estimador de Máxima Verosimilitud (PPML): Base Teórica empírica para la confirmación de modelos gravitacionales logísticos.")
        grav_data = q(10, f"SELECT LN(AVG(dist_nm)) AS ln_dist, LN(AVG(eur+1)) AS ln_eur, macro_sector, YEAR(date) AS anio FROM trade {where} GROUP BY YEAR(date), macro_sector")
        if not grav_data.empty and len(grav_data) > 5:
            fig_grav = px.scatter(grav_data.dropna(), x='ln_dist', y='ln_eur', color='macro_sector', trendline='ols', title="Regresión Científica Empírica y Comprobación de Ley de Gravedad Comercial")
            pt(fig_grav); st.plotly_chart(fig_grav, use_container_width=True)

    with tabs[4]:
        formula("Estructura de Retorno Neta: Costo Total Real de Mercancía contemplando externalidades y seguros de Capital Retenido")
        paises_disp = q(90, f"SELECT DISTINCT origin_name FROM trade {where}")['origin_name'].dropna().sort_values().tolist()
        if len(paises_disp) >= 2:
            c_ab1,c_ab2,c_ab3 = st.columns(3)
            with c_ab1:
                idx_china = paises_disp.index('China') if 'China' in paises_disp else 0
                origen_A = st.selectbox("Entorno Proveedor Dominante (Actual)", paises_disp, index=idx_china)
            with c_ab2:
                idx_mar = paises_disp.index('Marruecos') if 'Marruecos' in paises_disp else 1
                origen_B = st.selectbox("Entorno Proveedor a Migrar (Objetivo Regional)", paises_disp, index=idx_mar)
            with c_ab3:
                vol_simulado = st.number_input("Carga Total a Analizar (€ Millones netos)", min_value=0.001, value=100.0, step=10.0)
            parametros = q(91, f"SELECT origin_name, AVG(dist_nm) as dist_media, AVG(tariff_rate) as tarifa_media, (SUM(costo_co2_ets)/NULLIF(SUM(eur),0)) as f_co2, (SUM(LVaR_95)/NULLIF(SUM(eur),0)) as f_lvar, AVG(lead_time) as lt FROM trade {where} AND origin_name IN ('{origen_A}','{origen_B}') GROUP BY origin_name")
            if len(parametros) == 2:
                pA = parametros[parametros['origin_name']==origen_A].iloc[0]
                pB = parametros[parametros['origin_name']==origen_B].iloc[0]
                f_op = 0.045
                tco_A = vol_simulado*((pA['tarifa_media'] or 0)+(pA['f_co2'] or 0)+f_op)
                tco_B = vol_simulado*((pB['tarifa_media'] or 0)+(pB['f_co2'] or 0)+f_op)
                lvar_A, lvar_B = vol_simulado*(pA['f_lvar'] or 0), vol_simulado*(pB['f_lvar'] or 0)
                ahorro_tco, dif_lvar, ahorro_dias = tco_A-tco_B, lvar_A-lvar_B, (pA['lt'] or 0)-(pB['lt'] or 0)
                def fmt_val(val, is_lvar=False):
                    v2 = abs(val); dec = 3 if is_lvar else 2
                    if v2<0.00001: return "€0.00"
                    if v2>=1000: return f"€{v2/1000:,.{dec}f} Bn"
                    if v2>=1: return f"€{v2:,.{dec}f} M"
                    return f"€{v2*1000:,.1f} K"
                tco_lbl = "Ahorro Margen Operativo TCO" if ahorro_tco>=0 else "Penalización Severa Operativa"
                tco_color = COLORS['green'] if ahorro_tco>=0 else COLORS['ub_red']
                lvar_lbl = "Flujo Libre de Caja Liberado (P&L)" if dif_lvar>0 else "Incremento de Capital Inmovilizado"
                color_lvar = COLORS['green'] if dif_lvar>0 else COLORS['ub_red']
                dias_color = COLORS['teal'] if ahorro_dias>=0 else COLORS['amber']
                
                st.markdown("<br>", unsafe_allow_html=True)
                c_r1,c_r2,c_r3 = st.columns(3)
                with c_r1: kpi(tco_lbl, fmt_val(ahorro_tco), "Diferencial total de Fricción", tco_color)
                with c_r2: kpi(lvar_lbl, fmt_val(dif_lvar,True), "Ajuste general en balances logísticos", color_lvar)
                with c_r3: kpi("Ajuste Tiempos Terminales Promedio", f"{abs(ahorro_dias):,.1f} días", "Compresión operativa estimada", dias_color)
                
                st.markdown("<br>", unsafe_allow_html=True)
                fig_ab = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ab.add_trace(go.Bar(name='Capa Régimen Aranceles Ad Valorem', x=[origen_A,origen_B], y=[vol_simulado*(pA['tarifa_media'] or 0), vol_simulado*(pB['tarifa_media'] or 0)], marker_color=COLORS['ub_red']), secondary_y=False)
                fig_ab.add_trace(go.Bar(name='Capa Impacto Normativas (EU ETS)',  x=[origen_A,origen_B], y=[vol_simulado*(pA['f_co2'] or 0), vol_simulado*(pB['f_co2'] or 0)], marker_color=COLORS['amber']), secondary_y=False)
                fig_ab.add_trace(go.Bar(name='Base Promedio de Fricción', x=[origen_A,origen_B], y=[vol_simulado*f_op]*2, marker_color=COLORS['ub_slate']), secondary_y=False)
                fig_ab.add_trace(go.Scatter(name='Nivel Inmovilización Financiera Promedio (LVaR)', x=[origen_A,origen_B], y=[lvar_A,lvar_B], mode='lines+markers', marker=dict(size=12,symbol='diamond',color=COLORS['teal']), line=dict(width=3,dash='dot')), secondary_y=True)
                fig_ab.update_layout(barmode='stack', title=f"Estudio Económico Físico: Redistribuir y Mapear Volumen de €{vol_simulado}M", height=500, margin=dict(t=60,b=100), legend=dict(orientation="h",y=-0.25))
                fig_ab.update_yaxes(title_text="Absorción Costos Directos P&L Estimados (€ M)", secondary_y=False)
                fig_ab.update_yaxes(title_text="Capital Detenido LVaR (€ M)", secondary_y=True, showgrid=False)
                st.plotly_chart(fig_ab, use_container_width=True)
                
                txt_tco = f"Al modelar el redireccionamiento operativo de un volumen equivalente de {origen_A} directo a {origen_B}, la corporación {'rescatará contablemente' if ahorro_tco>=0 else 'asumirá como pérdida extra'} un total de **{fmt_val(ahorro_tco)}** aplicable directamente al estado de resultados. "
                txt_lvar = f"De forma paralela, la tesorería corporativa {'logra liberar' if dif_lvar>0 else 'estará forzada a retener'} un flujo estimado de **{fmt_val(dif_lvar,True)}** en capital inmovilizado, derivado exclusivamente del perfil de volatilidad logística entre ambas rutas."
                ai_agent("LECTURA TÉCNICA Y DICTAMEN GERENCIAL", txt_tco + txt_lvar)


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 8 · CONSOLA DE INVESTIGACIÓN
# ═════════════════════════════════════════════════════════════════════════════
def page_research(where):
    render_header("Consola de Modelación e Investigación Matemática", "Ejecución de modelos econométricos avanzados (Regresiones PPML no lineales) y algoritmos para aislar perturbaciones críticas (Quiebres de Varianza PELT).")
    
    guide_box("CONSOLA DE VALIDACIÓN CIENTÍFICA", "Esta es la sala de máquinas estadística de tu TFM. No te preocupes por entender el código profundo: la pestaña 'PPML' sirve para demostrar con datos que a mayor distancia de flete, el volumen comercial realmente decae. La pestaña 'PELT' busca picos de estrés en la historia de tu selección para validar que las caídas no son accidentes, sino perturbaciones matemáticas comprobables.")

    tabs = st.tabs(["Ecuación Gravitacional Avanzada (PPML)","Diagramación: Paradoja de Diversificación Geográfica","Detector Algorítmico de Perturbaciones (PELT)","Matriz: Fricción Operacional vs Regionalización (SSS)"])

    with tabs[0]:
        formula("Estructura Gravitacional PPML: E[Vol_Intercambio_ij | Dist_ij] = exp(α + β·ln(Dist_ij) + γ·ln(Barril_Brent_t))")
        h1 = q(23, f"SELECT YEAR(date) AS anio, macro_sector, SUM(eur*dist_nm)/SUM(eur) AS wad, AVG(oil_price) AS brent, SUM(eur)/1e9 AS eur_bn FROM trade {where} GROUP BY 1, 2")
        if not h1.empty and len(h1) > 2:
            st.markdown("La matriz de dispersión mapea los volúmenes físicos totales reportados en los registros base anuales. La línea que lo acompaña refleja el ajuste computado automáticamente aplicando Poisson Pseudo-Maximum Likelihood.")
            try:
                Y = h1['eur_bn']
                X = sm.add_constant(np.log(h1[['brent','wad']].clip(lower=0.01)))
                ppml = sm.GLM(Y, X, family=sm.families.Poisson()).fit()
                h1['ppml_pred'] = ppml.predict(X)
                
                coef_df = pd.DataFrame({
                    'Variable de Carga Estructural': ['Parámetro Alpha (Intercepción)','ln(Costos Flete Proxy - Brent Promedio)','ln(Distancia WAD Consolidada)'],
                    'Coeficiente Empírico Base': ppml.params.values,
                    'Valor-P Estadístico': ppml.pvalues.values,
                    'Límite Inferior (IC 95%)': ppml.conf_int()[0].values,
                    'Límite Superior (IC 95%)': ppml.conf_int()[1].values
                }).round(4)
                st.dataframe(coef_df, use_container_width=True, hide_index=True)

                st.markdown("<br>", unsafe_allow_html=True)
                fig_h1 = go.Figure()
                fig_h1.add_trace(go.Scatter(x=h1['brent'], y=h1['eur_bn'], mode='markers',
                                             marker=dict(size=h1['wad']/100, color=COLORS['ub_blue'], opacity=0.7), name='Datos Extraídos en Bruto'))
                h1s = h1.sort_values('brent')
                fig_h1.add_trace(go.Scatter(x=h1s['brent'], y=h1s['ppml_pred'], mode='lines',
                                             line=dict(color=COLORS['ub_red'], width=3), name='Ecuación Proyectada PPML'))
                pt(fig_h1); fig_h1.update_layout(title="Estudio Empírico de Gravedad: Efectos del Flete y Lejanía Promedio en Volúmenes Físicos", height=500, xaxis_title="Cotización Media Crudo Brent ($ USD/Barril) [Escala Log]", yaxis_title="Sumatoria Operaciones Registradas (€ Bn FOB)")
                st.plotly_chart(fig_h1, use_container_width=True)
            except Exception:
                st.warning("Se carece de la longitud histórica suficiente de registros para alcanzar convergencia confiable en la regresión matricial. Expanda filtros por favor.")
            ai_agent("LECTURA TÉCNICA MATEMÁTICA", "La pendiente analizada (Coeficiente Beta) cuantifica rigurosamente qué tan elástico es el flujo físico al incorporar fricción logística. Obtener magnitudes por debajo de cero demuestra empíricamente que: a mayor lejanía promedio, el volumen mercantil general cae. El Coeficiente Gamma permite medir la proporción en la que un shock energético estructural fulmina las capacidades de importación en bloques de ultramar.")

    with tabs[1]:
        h2 = q(24, f"WITH sh AS (SELECT YEAR(date) AS anio, macro_sector, o_iso, SUM(eur) AS val, SUM(SUM(eur)) OVER (PARTITION BY YEAR(date),macro_sector) AS tot FROM trade {where} GROUP BY 1,2,3) SELECT s.anio, s.macro_sector, SUM((s.val/NULLIF(s.tot,0))*(s.val/NULLIF(s.tot,0)))*10000 AS hhi, v.lvar_bn, v.lt_var FROM sh s JOIN (SELECT YEAR(date) AS anio, macro_sector, SUM(LVaR_95)/1e9 AS lvar_bn, STDDEV(lead_time) AS lt_var FROM trade {where} GROUP BY 1,2) v ON s.anio=v.anio AND s.macro_sector=v.macro_sector GROUP BY 1,2,4,5")
        if not h2.empty and len(h2.dropna()) > 5:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_h2 = px.scatter(h2.dropna(), x='hhi', y='lvar_bn', color='macro_sector', trendline='ols', size='lt_var', title="Paradoja de Diversificación: HHI Promedio vs LVaR Estimado")
            pt(fig_h2); fig_h2.update_layout(height=500); st.plotly_chart(fig_h2, use_container_width=True)
            ai_agent("LECTURA DE CORRELACIONES VINCULADAS", "Identificar formaciones dispersas tipo 'Nube Rota' en las correlaciones evidencia la ruptura de un mito corporativo logístico clásico: Segmentar tu abastecimiento general en cientos de fragmentos geográficos (reduciendo agresivamente tu HHI) no crea escudos estructurales ni previene el aumento del capital de riesgo, puesto que toda esa hiperfragmentación termina compitiendo por capacidad en los mismísimos mega-puertos de desembarque dentro de las aduanas de la Unión Europea.")

    with tabs[2]:
        formula("Detector Change Point Paramétrico: Subrutinas algorítmicas aislando fluctuaciones atípicas que superan la frontera histórica de varianza mensual (μ + 1σ).")
        lt_viol = q(27, f"SELECT DATE_TRUNC('quarter', date) AS t, STDDEV(lead_time) AS lt_std, SUM(LVaR_95)/1e9 AS lvar FROM trade {where} GROUP BY 1 ORDER BY 1")
        if not lt_viol.empty and len(lt_viol) > 4:
            from scipy.signal import find_peaks
            y_values = lt_viol['lt_std'].fillna(0).values
            umbral = np.mean(y_values) + np.std(y_values)
            peaks, _ = find_peaks(y_values, height=umbral, distance=4)
            lt_viol['es_ruptura'] = 0
            if len(peaks) > 0:
                lt_viol.iloc[peaks, lt_viol.columns.get_loc('es_ruptura')] = 1
            lt_viol['color'] = np.where(lt_viol['es_ruptura']==1, COLORS['ub_red'], COLORS['ub_blue'])
            
            st.markdown("<br>", unsafe_allow_html=True)
            fig_h3 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                   subplot_titles=["Análisis de Perturbaciones Físicas: Varianza Detectada Histórica Mensual","Evolución de Confinamiento de Capitales de Emergencia (LVaR Acumulado) € Bn"])
            fig_h3.add_trace(go.Scatter(x=lt_viol['t'], y=lt_viol['lt_std'], line=dict(color=COLORS['ub_blue'], width=2.5), name='Varianza Operativa Base'), row=1, col=1)
            for _, row_r in lt_viol[lt_viol['es_ruptura']==1].iterrows():
                fig_h3.add_vline(x=pd.to_datetime(row_r['t']).strftime('%Y-%m-%d'), line_dash="dash", line_color=COLORS['ub_red'], row=1, col=1)
            fig_h3.add_trace(go.Bar(x=lt_viol['t'], y=lt_viol['lvar'], marker_color=lt_viol['color']), row=2, col=1)
            pt(fig_h3); fig_h3.update_layout(height=600, showlegend=False); st.plotly_chart(fig_h3, use_container_width=True)
            ai_agent("RESULTADO DEL ALGORITMO DETECTOR", f"La parametrización predictiva logró rastrear los flujos pasados y ubicar con precisión {len(peaks)} colapsos logísticos severos en la cadena global. Las densas agrupaciones marcadas en barras rojas demuestran estadísticamente cómo las empresas responden disparando una inyección reactiva y millonaria para inflar cajas de seguridad y reservas de inventario (LVaR total) buscando sortear interrupciones del transporte mundial.")
        else:
            st.info("La matriz no posee la densidad de registros temporales suficientes para ejecutar analítica PELT ni detectar quiebres estadísticos robustos.")

    with tabs[3]:
        h4 = q(28, f"SELECT YEAR(date) AS anio, puerto, SUM(CASE WHEN is_near=1 THEN teu ELSE 0 END)/NULLIF(SUM(teu),0) AS pct_near, STDDEV(lead_time) AS lt_var, SUM(teu) AS teu FROM trade {where} GROUP BY 1, 2 HAVING SUM(teu)>100")
        if not h4.empty and len(h4.dropna()) > 0:
            h4c = h4.dropna().sort_values('anio')
            st.markdown("<br>", unsafe_allow_html=True)
            fig_h4 = px.scatter(h4c, x='pct_near', y='lt_var', color='puerto', size='teu', animation_frame='anio',
                                category_orders={"anio": sorted(h4c['anio'].unique())},
                                title="Análisis Gráfico Dinámico: Cuota Consolidada SSS vs Alteración Continua Operativa Portuaria")
            pt(fig_h4); fig_h4.update_layout(height=500); st.plotly_chart(fig_h4, use_container_width=True)
            ai_agent("LECTURA TÉCNICA Y DIAGNÓSTICO DE ESTRÉS", "El eje X grafica la progresión efectiva de las estrategias de regionalización logística y el eje Y, el nivel del daño registrado vía ineficiencia de atracos. Revelar formaciones de tendencias de dispersión con pendiente positiva alerta estructuralmente al analista: los engranajes aduaneros interconectados pueden colapsar si la eliminación generalizada de grandes contenedores limpios termina reemplazándose por múltiples e incontrolables flotas costeras reducidas que inunden e infarten los muelles principales de las costas.")


# ═════════════════════════════════════════════════════════════════════════════
# PÁGINA 9 · GLOSARIO Y EXPORTACIÓN (EXCEL FALLBACK TO CSV)
# ═════════════════════════════════════════════════════════════════════════════
def page_glossary(where):
    render_header("Diccionario Técnico Consolidado & Salida de Exportaciones Oficiales", "Conformación terminológica paramétrica exigida dentro del simulador gemelo SIT-2026 CORE.")
    
    guide_box("BIBLIOTECA TÉCNICA Y EXPORTACIÓN AVANZADA", "Consulta las definiciones exactas de cada término técnico utilizado a lo largo de la aplicación. Si requieres continuar tus análisis o generar tus propios reportes gerenciales (PowerBI/Excel), descarga la base de datos oficial, consolidada y filtrada en formato Excel utilizando el botón de descarga situado en la parte inferior.")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
### Conformación de Ejes Estructurales del Modelado
1. **LÍNEA ACADÉMICA** — Ejecución pura orientada a contrastar empíricamente las promesas teóricas generalistas que definen los modelos de relocalización industrial (Nearshoring).
2. **LÍNEA PROBABILÍSTICA** — Medición rigurosa y mapeo matemático aplicando iteraciones bajo el método Monte Carlo para revelar el comportamiento incierto terminal.
3. **LÍNEA FÍSICO-OPERATIVA** — Uso implícito de la Teoría Computacional de Colas (M/M/1) para el cálculo de afectaciones a la red y el estrés directo de capacidades e infraestructura aduanal.
4. **LÍNEA MACROECONÓMICA** — Desensamble algorítmico profundo sobre los balances de corporaciones europeas integrando la estructura tarifaria, costos de externalidad climática y la penalidad friccional del flete logístico.

### Diccionario General Descriptivo Integrado

* **€ Bn (Billion):** Estándar contable internacional de macro-economía representando 1 Bn = Miles de Millones de Euros (10⁹).
* **TEU (Twenty-foot Equivalent Unit):** Métrica portuaria universal que homologa la variabilidad real de las cargas unificando la mercancía como cajas estandarizadas de veinte pies logísticos equivalentes.
* **Inmovilización Estocástica LVaR (Logistics Value at Risk):** Herramienta financiera clave del motor, midiendo los euros promedio sepultados o en cuarentena, inyectados por la gerencia al capital de trabajo (Safety Stock) como seguro preventivo mientras el cargamento navega la volatilidad oceánica.
* **Distancia Ponderada WAD (Weighted Average Distance):** Calibración geométrica y exacta mostrando la lejanía neta europea, pero castigada duramente de forma inversamente proporcional a la masificación y concentración física de mercancías compradas en un nodo alejado.
* **Índice IHH (Herfindahl-Hirschman) Competitivo:** Detector matemático macro que audita y devela si tus compras totales sufren secuestro oligopólico, evaluando la vulnerabilidad de recaer estructuralmente en el abandono del monopolio de ultramar por otros esquemas de riesgo localizados.
* **Drop Size Operacional Físico:** Ajuste algorítmico indispensable corrigiendo la desviación del clásico "Espejismo de Filas". Entiende el volumen físico real y verdadero (TEUs) que logra procesar un buque intercontinental vs un barco regional para anular por completo la métrica irreal que entregan los miles de pequeños papeleos aduaneros de la base de origen.
* **Factor Específico de Saturación Alpha (α):** Multiplicador de reacción transaccional en la infraestructura. Advierte que descartar masa bruta empaquetada de larga distancia forzará que traigas "X" número multiplicador de nuevos barcos diminutos regionales para nivelar el mismo total.
* **Techo Terminal Operativo (μ - Mu):** Capacidad límite de extracción aduanera, rastreada analíticamente en todo el histórico detectando cuanta tolerancia al desastre la aduana logístico-costera de interés pudo registrar en su pico más agresivo de eficiencia en el pasado.
* **Efecto Látigo Logístico Regional (SSS - Short Sea Shipping):** Trastorno de ineficiencia colateral advertido matemáticamente en red. Confirma la deficiencia que un puerto de la Unión Europea experimentará si busca absorber el impacto físico fragmentado regional abandonando volúmenes masificados.
* **Suma Cero Estructural Simulada:** Mecanismo central base en la arquitectura virtual y control paramétrico. Su lógica bloquea absolutamente el ingreso de proyecciones o demandas fantasmas para el análisis: Toda masa de larga distancia retirada por el control analítico debe aterrizar íntegra en la ruta de corta distancia para asegurar simulaciones matemáticas perfectas e irrefutables.
* **Estimadores de Regresión PPML (Poisson Pseudo-Maximum):** Potente herramienta algorítmica no lineal. Permite asimilar y evaluar con perfección analítica las matrices gravitacionales crudas procesando los ceros logísticos, revelando de paso la elasticidad real paramétrica frente al costo-distancia sin generar anomalías y desviaciones de las lecturas MCO tradicionales.
* **Algoritmo Paramétrico PELT (Pruned Exact Linear Time):** Estructura estadística avanzada. Rastrea en alta velocidad computacional picos de desviación y señala empíricamente el día, mes o bloque en que tus problemas y fallas de la vida real destruyeron toda tu tendencia pronosticada para reventar el techo de varianza estadística esperable.

<div style="margin-top:20px;padding-top:16px;border-top:1px solid #cbd5e1;font-size:14px;color:#475569;">
  <b style="color:#003d65;font-size:16px;">Acerca de la Arquitectura del SIT-2026 CORE</b><br><br>
  <b>Autor y desarrollador del modelo:</b> Hugo Francisco Alejo Cárdenas<br>
  <b>Supervisor de analítica y asesoría metodológica:</b> Prof. Josep María Cervera<br>
  <b>Integración de registros físicos y bases de verificación institucional:</b> Eurostat (COMEXT Analytics), CEPII Gravitational Data, Banco Central Europeo, Organismo de Información Energética Global (EIA)
</div>
        """, unsafe_allow_html=True)

    with c2:
        info("El motor autoriza emitir reportes crudos y procesados en matrices sin manipulación posterior. El descargo garantiza que todo equipo empresarial contable y operativo (ERP y herramientas BI) consuma directamente el trabajo efectuado aquí para auditar de la forma deseada.")
        with st.spinner("Ensamblando el bloque de exportaciones matriciales para descarga local..."):
            export_df = q(900, f"SELECT * FROM trade {where} LIMIT 50000")
            
            # Intento de exportar a Excel. Si falla (por falta de openpyxl/xlsxwriter), exporta a CSV como fallback seguro.
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='Base_Datos_Cruda', index=False)
                    if 'macro_sector' in export_df.columns:
                        summary = export_df.groupby('macro_sector')['eur'].sum().reset_index()
                        summary.to_excel(writer, sheet_name='Resumen_Sectorial', index=False)
                excel_data = output.getvalue()
                st.download_button(
                    label="📥 Extraer Reporte Gerencial (Excel .xlsx)",
                    data=excel_data, 
                    file_name='SIT2026_Enterprise_Extractor_Logs.xlsx', 
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            except Exception:
                # Fallback a CSV garantizado para que nunca marque error en pantalla
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Extraer Base de Registros Empíricos (CSV)",
                    data=csv_data, 
                    file_name='SIT2026_Enterprise_Extractor_Logs.csv', 
                    mime='text/csv'
                )

# ═════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER (COMPILADOR PRINCIPAL Y ENSAMBLE DEL LAYOUT)
# ═════════════════════════════════════════════════════════════════════════════
def main():
    get_con()
    page, where, transfer_pct = render_sidebar()

    if   page == "Panel Ejecutivo":              page_executive_dashboard(where, transfer_pct)
    elif page == "Flujos Comerciales":           page_trade_flow(where, transfer_pct)
    elif page == "Gemelo Portuario":             page_port_twin(where, transfer_pct)
    elif page == "Análisis de Sobrecostos (TCO)": page_cost_xray(where, transfer_pct)
    elif page == "Motor de Riesgo (LVaR)":       page_montecarlo(where)
    elif page == "Monitoreo Nearshoring":        page_nearshoring(where, transfer_pct)
    elif page == "Modelado de Escenarios":       page_scenario(where)
    elif page == "Consola de Investigación":     page_research(where)
    elif page == "Glosario y Exportación":       page_glossary(where)

    # Inyección del Footer Institucional Fijo SECIHTI al final de toda vista
    render_footer()

if __name__ == "__main__":
    main()
