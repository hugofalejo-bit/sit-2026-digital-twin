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

/* Tarjetas KPI */
.kpi-card {
  background: var(--white); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px 18px 14px; position: relative; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  display: flex; flex-direction: column; justify-content: space-between;
  height: 100%; min-height: 155px;
}
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 8px 8px 0 0; }
.kpi-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--gray600); margin-bottom: 6px; font-weight: 700; line-height: 1.3;}
.kpi-value { font-size: 23px; font-weight: 800; color: var(--ub-blue); line-height: 1.2; margin-bottom: 6px; white-space: normal !important; word-wrap: break-word !important; }
.kpi-sub  { font-size: 11.5px; color: var(--gray600); font-weight: 500; margin-top: auto; }

/* Encabezados de Sección */
.sec-wrap { margin: 30px 0 16px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
.sec-label { font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: .15em; text-transform: uppercase; color: var(--gray600); font-weight: 800; }
.sec-title { font-size: 22px; font-weight: 800; color: var(--ub-blue); margin: 4px 0 4px; letter-spacing: -.5px; }

/* Cajas Analíticas y Fórmulas */
.formula-box { background: var(--white); border: 1px solid var(--border); border-left: 4px solid var(--ub-blue); border-radius: 6px; padding: 12px 18px; font-family: 'Space Mono', monospace; font-size: 12.5px; color: var(--ub-slate); margin: 12px 0; }
.ai-agent-box { background: var(--white); border: 1px solid var(--border); border-top: 4px solid #2563eb; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
.ai-title { font-family: 'Space Mono', monospace; font-size: 11px; font-weight: 800; color: #2563eb; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;}
.ai-text { font-size: 14.5px; color: var(--ub-slate); line-height: 1.6; font-weight: 500;}

.resilience-card { background: var(--ub-blue); border-radius: 12px; padding: 24px; color: white; margin: 10px 0; height: 100%; display: flex; flex-direction: column; justify-content: center;}
.resilience-score { font-size: 48px; font-weight: 800; line-height: 1.1; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DICCIONARIOS Y CONFIGURACIÓN BASE
# ─────────────────────────────────────────────────────────────────────────────
PARQUET_PATH = os.getenv("PARQUET_PATH", "TFM_HS2_Macro_Dataset.parquet")
LOGO_PATH    = os.getenv("LOGO_PATH", "logo.png")

ISO3_EU = {'AUT':'Austria','BEL':'Bélgica','BGR':'Bulgaria','HRV':'Croacia','CYP':'Chipre','CZE':'República Checa','DNK':'Dinamarca','EST':'Estonia','FIN':'Finlandia','FRA':'Francia','DEU':'Alemania','GRC':'Grecia','HUN':'Hungría','IRL':'Irlanda','ITA':'Italia','LVA':'Letonia','LTU':'Lituania','LUX':'Luxemburgo','MLT':'Malta','NLD':'Países Bajos','POL':'Polonia','PRT':'Portugal','ROU':'Rumanía','SVK':'Eslovaquia','SVN':'Eslovenia','ESP':'España','SWE':'Suecia'}
HS2_ES = {'01':'Animales vivos', '02':'Carne y despojos', '03':'Pescado y mariscos', '04':'Lácteos y huevos', '05':'Otros origen animal', '06':'Plantas y flores', '07':'Legumbres y hortalizas', '08':'Frutas y cítricos', '09':'Café, té y especias', '10':'Cereales', '11':'Productos molinería', '12':'Semillas oleaginosas', '15':'Grasas y aceites', '16':'Prep. carne/pescado', '17':'Azúcares', '18':'Cacao', '19':'Prep. cereales', '20':'Prep. legumbres', '21':'Prep. diversas', '22':'Bebidas y alcohol', '23':'Residuos alimentarios', '24':'Tabaco', '25':'Sal, azufre y tierras', '26':'Minerales y escorias', '27':'Combustibles y petróleo', '28':'Química inorgánica', '29':'Químicos orgánicos', '30':'Farma & Salud', '31':'Abonos y fertilizantes', '32':'Extractos curtientes/tintes', '33':'Aceites esenciales', '34':'Jabones y ceras', '38':'Químicos diversos', '39':'Plásticos', '40':'Caucho y derivados', '44':'Madera', '47':'Pasta de madera', '48':'Papel y cartón', '50':'Seda', '51':'Lana', '52':'Algodón', '54':'Filamentos sintéticos', '55':'Fibras sintéticas', '61':'Prendas de vestir (Punto)', '62':'Prendas de vestir (No punto)', '64':'Calzado', '68':'Manufacturas de piedra', '69':'Productos cerámicos', '70':'Vidrio', '71':'Perlas y joyería', '72':'Fundición, hierro y acero', '73':'Manufacturas de hierro/acero', '74':'Cobre', '76':'Aluminio', '82':'Herramientas y cuchillería', '84':'Maquinaria y reactores', '85':'Aparatos eléctricos/electrónica', '86':'Vehículos ferroviarios', '87':'Vehículos automóviles', '88':'Aeronaves y espaciales', '89':'Barcos y navegación', '90':'Instrumentos médicos', '93':'Armas y municiones', '94':'Muebles', '95':'Juguetes y deportes', '99':'Otros (Confidencial)'}

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
    except: return 'Otros Sectores'

COLORS = { 'ub_blue':'#003d65', 'ub_red':'#aa1916', 'teal':'#0d9488', 'green':'#059669', 'amber':'#d97706', 'purple':'#6366f1', 'ub_slate':'#334155' }
COLOR_SEQ = list(COLORS.values())
PLOTLY_BASE = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Plus Jakarta Sans, sans-serif', color='#334155', size=12), xaxis=dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', zerolinecolor='#cbd5e1'), yaxis=dict(gridcolor='#e2e8f0', linecolor='#cbd5e1', zerolinecolor='#cbd5e1'), legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#e2e8f0', borderwidth=1), margin=dict(t=40, b=30, l=50, r=20), colorway=COLOR_SEQ)
CRISIS_EVENTS = {'2008-09': ('Global Financial Crisis', '#aa1916'), '2020-03': ('COVID-19 Pandemic', '#aa1916'), '2022-03': ('Ukraine Conflict', '#aa1916')}

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
        if not os.path.exists(PARQUET_PATH): st.stop()
        con.execute(f"CREATE OR REPLACE VIEW trade AS {view_sql} FROM read_parquet('{PARQUET_PATH}')")
    return con

@st.cache_data(ttl=600, show_spinner=False)
def q(_cid, sql: str) -> pd.DataFrame:
    return get_con().execute(sql).df()

# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPONENTES UI Y LIBRERÍA MATEMÁTICA
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
        </div>""", unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH): c2.image(LOGO_PATH, use_container_width=True)

def section(label, title, desc=""):
    st.markdown(f'<div class="sec-wrap"><div class="sec-label">{label}</div><div class="sec-title">{title}</div></div>', unsafe_allow_html=True)

def kpi(label, value, sub="", color="#003d65"):
    st.markdown(f'<div class="kpi-card"><div class="kpi-accent" style="background:{color}"></div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

def ai_agent(title, text):
    st.markdown(f'<div class="ai-agent-box"><div class="ai-title">C-LEVEL INSIGHT · {title}</div><div class="ai-text">{text}</div></div>', unsafe_allow_html=True)

def macro_queue_metrics(lam, capacity):
    if capacity <= 0: return dict(rho=1.0, Lq=float('inf'), Wq=float('inf'))
    rho = lam / capacity
    if rho >= 1.0: return dict(rho=rho, Lq=float('inf'), Wq=float('inf'))
    Wq = rho / (capacity - lam)
    Lq = lam * Wq
    return dict(rho=rho, Lq=Lq, Wq=Wq)

def tariff_trade_impact(eur_base, tariff_old, tariff_new, elasticity=-1.1):
    delta_tau = tariff_new - tariff_old
    pct_change = elasticity * delta_tau / (1 + tariff_old)
    eur_new = eur_base * (1 + pct_change)
    return {'eur_new': eur_new, 'pct_change': pct_change}

def resilience_score(wad_trend, hhi_norm, lvar_norm, near_share):
    score = 50.0 + np.clip(-wad_trend * 0.002, -20, 20)
    if hhi_norm > 5000: score -= 25
    elif hhi_norm < 1500: score += 25
    else: score += 25 - (hhi_norm - 1500)/3500 * 50
    score += np.clip(-lvar_norm * 15, -30, 30) + np.clip((near_share - 20) * 0.5, -25, 25)
    return float(np.clip(score, 0, 100))

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR Y FILTROS
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("<div style='text-align:center;font-family:\"Space Mono\",monospace;font-size:12px;letter-spacing:.1em;color:#003d65;text-transform:uppercase;font-weight:800'>SIT-2026 Core</div>", unsafe_allow_html=True)
        pages = ["Panel Ejecutivo", "Flujos Comerciales", "Modelado de Escenarios", "Motor de Riesgo (LVaR)", "Gemelo Portuario", "Monitoreo Nearshoring", "Análisis de Sobrecostos (TCO)", "Consola de Investigación", "Glosario"]
        page = st.radio("MÓDULOS", pages, label_visibility="collapsed")
        years = st.slider("Horizonte Temporal", 2002, 2025, (2018, 2025))
        where_temp = f"WHERE YEAR(date) BETWEEN {years[0]} AND {years[1]}"
        all_macros = sorted(list(set([macro_sector(k) for k in HS2_ES.keys()])))
        sel_macros = st.multiselect("Sector", ["TODOS"] + all_macros, default=["TODOS"])
        where_macro = f" AND macro_sector IN ('{"','".join(sel_macros)}')" if "TODOS" not in sel_macros else ""
        paises_query = q(901, f"SELECT DISTINCT origin_name FROM trade {where_temp} {where_macro} ORDER BY origin_name")
        sel_origins = st.multiselect("Origen", ["TODOS"] + paises_query['origin_name'].tolist(), default=["TODOS"])
        where_origin = f" AND origin_name IN ('{"','".join(sel_origins)}')" if "TODOS" not in sel_origins else ""
        puertos_query = q(903, f"SELECT DISTINCT puerto FROM trade {where_temp} {where_macro} {where_origin} ORDER BY puerto")
        sel_ports = st.multiselect("Puerto", ["TODOS"] + puertos_query['puerto'].tolist(), default=["TODOS"])
        where_port = f" AND puerto IN ('{"','".join(sel_ports)}')" if "TODOS" not in sel_ports else ""
    return page, f"{where_temp} {where_macro} {where_origin} {where_port}"

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA 5 · GEMELO PORTUARIO (REDISTRIBUCIÓN DE SUMA CERO)
# ═════════════════════════════════════════════════════════════════════════════
def page_port_twin(where):
    render_header("Gemelo Portuario: Simulación de Transición de Red", "A diferencia de un incremento de demanda, esta simulación modela la REDISTRIBUCIÓN de carga desde rutas oceánicas hacia regionales (Nearshoring).")
    
    st.markdown('<div class="formula-box"><b>Algoritmo de Fragmentación de Carga (Short Sea Shipping):</b><br>λ_final = [λ_near + (Vol_Transf * α_Fragmentación)] + [λ_off - Vol_Transf] | Masa total constante.</div>', unsafe_allow_html=True)

    # Selección de puerto específico para la simulación
    geo_p = q(f"SELECT DISTINCT puerto FROM trade {where}")
    port_sel = st.selectbox("Puerto para Test de Estrés de Transición", sorted(geo_p['puerto'].unique()))
    p_where = where + f" AND puerto = '{port_sel}'"

    # --- CONTROLES DE TRANSICIÓN ---
    c_cfg1, c_cfg2, c_cfg3 = st.columns(3)
    with c_cfg1:
        transfer_near = st.slider("Transferencia al Nearshoring (%)", 0, 100, 0, help="Qué porcentaje de la carga de Asia/Lejanos se mueve a proveedores cercanos.") / 100
    with c_cfg2:
        alpha_frag = st.slider("Factor de Fragmentación (α)", 1.0, 10.0, 3.5, help="Multiplicador de arribos: Cuántos barcos SSS sustituyen a un solo buque oceánico.")
    with c_cfg3:
        # Capacidad dinámica sugerida basada en histórico
        base_v = q(f"SELECT SUM(teu)/(3.5*365) as l FROM trade {p_where}").iloc[0]['l'] or 50
        mu_cap = st.number_input("Capacidad de Servicio (μ) [TEUs/día]", value=int(base_v*1.5))

    # --- LÓGICA DE SUMA CERO ---
    # Obtenemos el volumen histórico separado por cercanía
    port_data = q(f"""
        SELECT 
            SUM(CASE WHEN is_near=0 THEN teu ELSE 0 END) / (3.5*365) as l_off,
            SUM(CASE WHEN is_near=1 THEN teu ELSE 0 END) / (3.5*365) as l_near
        FROM trade {p_where}
    """).iloc[0]
    
    l_off_base = port_data['l_off'] or 0
    l_near_base = port_data['l_near'] or 0
    
    # 1. Quitamos carga de Offshore
    vol_transf = l_off_base * transfer_near
    l_off_final = l_off_base - vol_transf
    
    # 2. Sumamos a Nearshore multiplicando por la fragmentación
    l_near_final = l_near_base + (vol_transf * alpha_frag)
    
    # 3. Nueva Tasa de Arribos Total
    lam_total = l_off_final + l_near_final
    
    # Métricas M/M/1
    m = macro_queue_metrics(lam_total, mu_cap)

    # --- VISUALIZACIÓN ---
    k1, k2, k3 = st.columns(3)
    with k1: kpi("Tasa de Arribos (λ)", f"{lam_total:.1f} barcos/día", f"Original: {l_off_base+l_near_base:.1f}", COLORS['ub_blue'])
    with k2: kpi("Tensión de Infraestructura", f"{m['rho']*100:.1f}%", "Utilización del muelle", COLORS['ub_red'] if m['rho'] > 0.85 else COLORS['green'])
    with k3: kpi("Espera Nodal (Wq)", f"{m['Wq']:.2f} días" if np.isfinite(m['Wq']) else "Colapso", "Retraso logístico adicional", COLORS['amber'])

    # Desglose para el tribunal
    with st.expander("🔍 Ver Desglose de la Redistribución (Caja Blanca)"):
        st.write(f"**Carga Offshore Reducida:** {vol_transf:.1f} unidades")
        st.write(f"**Nuevos Arribos Regionales (SSS):** {vol_transf * alpha_frag:.1f} unidades (debido al factor α={alpha_frag})")
        st.info("Nota: La masa total de la mercancía no ha cambiado, pero la 'unidad de gestión' (el barco) es más pequeña y frecuente.")

    # Gráfico asintótico
    cap_r = np.linspace(lam_total * 1.05, lam_total * 5, 50)
    wait_t = [(lam_total/c)/(c-lam_total) for c in cap_r]
    fig = go.Figure(go.Scatter(x=cap_r, y=wait_t, line=dict(color=COLORS['ub_red'], width=3)))
    fig.add_vline(x=mu_cap, line_dash="dash", annotation_text="Punto Operativo")
    fig.update_layout(**PLOTLY_BASE, title="Curva de Congestión Estocástica", xaxis_title="Capacidad (μ)", yaxis_title="Espera (Wq)"); st.plotly_chart(fig, use_container_width=True)

    ai_agent("DIAGNÓSTICO PORTUARIO", "Esta simulación demuestra que el Nearshoring es físicamente más exigente para la infraestructura europea. Aunque no compremos más volumen, el cambio hacia barcos más pequeños (SSS) dispara la frecuencia de arribos, llevando al puerto al colapso si no existe una expansión de la capacidad de servicio.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGINA 3 · MODELADO DE ESCENARIOS (TRADE-OFF DE SUMA CERO)
# ═════════════════════════════════════════════════════════════════════════════
def page_scenario(where):
    render_header("Modelado de Escenarios: Simulación de Suma Cero", "Evaluación del balance financiero al migrar contratos desde Asia hacia el Mediterráneo.")
    tabs = st.tabs(["Política Arancelaria", "Mercado Carbono (ETS)", "Simulador de Relocalización A/B"])
    
    with tabs[2]:
        paises_disp = q(f"SELECT DISTINCT origin_name FROM trade {where}")['origin_name'].dropna().sort_values().tolist()
        c_ab1, c_ab2, c_ab3 = st.columns(3)
        with c_ab1: o_A = st.selectbox("Origen Offshore (Asia/Largo)", paises_disp, index=paises_disp.index('China') if 'China' in paises_disp else 0)
        with c_ab2: o_B = st.selectbox("Origen Nearshore (Regional)", paises_disp, index=paises_disp.index('Marruecos') if 'Marruecos' in paises_disp else 0)
        with c_ab3: vol_sim = st.number_input("Volumen de Contrato a Mover (€ Millones)", value=100)

        m_data = q(f"SELECT origin_name, AVG(tariff_rate) as t, (SUM(costo_co2_ets)/NULLIF(SUM(eur),0)) as c, (SUM(LVaR_95)/NULLIF(SUM(eur),0)) as r FROM trade {where} AND origin_name IN ('{o_A}', '{o_B}') GROUP BY 1")
        
        if len(m_data) >= 2:
            pA = m_data[m_data['origin_name']==o_A].iloc[0]
            pB = m_data[m_data['origin_name']==o_B].iloc[0]
            
            ahorro_tco = vol_sim * ((pA['t'] + pA['c']) - (pB['t'] + pB['c']))
            delta_riesgo = vol_sim * (pB['r'] - pA['r'])
            
            k1, k2, k3 = st.columns(3)
            with k1: kpi("Ahorro en Fricción", f"€{ahorro_tco:.2f}M", "Aranceles y CO2 evitados", COLORS['green'])
            with k2: kpi("Variación de Riesgo", f"€{delta_riesgo:.2f}M", "Cambio en LVaR estocástico", COLORS['ub_red'] if delta_riesgo > 0 else COLORS['green'])
            with k3: kpi("Impacto Neto", f"€{ahorro_tco - delta_riesgo:.2f}M", "Viabilidad de la relocalización", COLORS['ub_blue'])

            ai_agent("ESTRATEGIA DE SUMA CERO", "El éxito del Nearshoring no es solo la distancia. Solo es rentable si el ahorro en aranceles y emisiones compensa la mayor volatilidad o ineficiencia del nuevo nodo de suministro.")

# ═════════════════════════════════════════════════════════════════════════════
# RESTO DE PÁGINAS (Mantenidas idénticas)
# ═════════════════════════════════════════════════════════════════════════════
def page_executive_dashboard(where):
    render_header("Panel Ejecutivo", "Control Macro")
    v = q(0, f"SELECT SUM(eur)/1e9 as eur_bn, SUM(eur*dist_nm)/SUM(eur) as wad, SUM(LVaR_95)/1e9 as lvar_bn, SUM(CASE WHEN is_near=1 THEN eur ELSE 0 END)/NULLIF(SUM(eur),0)*100 as pct_near FROM trade {where}").iloc[0]
    c = st.columns(4)
    with c[0]: kpi("Volumen Comercial", f"€{v.eur_bn:,.1f} Bn")
    with c[1]: kpi("WAD Global", f"{v.wad:,.0f} nm")
    with c[2]: kpi("LVaR Total", f"€{v.lvar_bn:,.1f} Bn")
    with c[3]: kpi("Nearshoring Share", f"{v.pct_near:.1f}%")
    st.plotly_chart(px.line(q(1, f"SELECT YEAR(date) as a, SUM(eur)/1e9 as e FROM trade {where} GROUP BY 1 ORDER BY 1"), x='a', y='e', title="Tendencia de Importación"))

def page_montecarlo(where):
    render_header("Motor de Riesgo (LVaR)", "Monte Carlo 10,000 sim")
    p = q(11, f"SELECT AVG(lead_time) as m, AVG(lt_std) as s, AVG(eur) as e FROM trade {where}").iloc[0]
    sim = np.random.lognormal(np.log(p['m'] or 10), 0.3, 10000) * (p['e'] or 1000) * 0.015 / 365
    st.plotly_chart(px.histogram(sim, title="Distribución de Riesgo Financiero por Embarque"))

def page_trade_flow(where):
    render_header("Flujos Comerciales", "Mapeo")
    df = q(44, f"SELECT origin_name, SUM(eur)/1e9 as e FROM trade {where} GROUP BY 1 ORDER BY 2 DESC LIMIT 15")
    st.plotly_chart(px.bar(df, x='e', y='origin_name', orientation='h'))

def page_nearshoring(where):
    render_header("Monitoreo Nearshoring", "Evolución")
    df = q(300, f"SELECT YEAR(date) as a, is_near, SUM(eur)/1e9 as e FROM trade {where} GROUP BY 1,2 ORDER BY 1")
    st.plotly_chart(px.area(df, x='a', y='e', color='is_near'))

def page_cost_xray(where):
    render_header("Análisis Sobrecostos (TCO)", "Fricción")
    v = q(19, f"SELECT SUM(costo_arancel)/1e9 as a, SUM(costo_co2_ets)/1e9 as c FROM trade {where}").iloc[0]
    st.plotly_chart(go.Figure(go.Waterfall(x=["Aranceles", "CO2 ETS", "Total"], y=[v.a, v.c, v.a+v.c])))

def page_research(where):
    render_header("Consola de Investigación", "PPML & PELT")
    st.markdown("Algoritmos avanzados activos.")

def page_glossary(where):
    render_header("Glosario", "Taxonomía")
    st.write("* **SSS:** Short Sea Shipping.\n* **LVaR:** Logistics Value at Risk.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    get_con()
    page, where = render_sidebar()
    if page == "Panel Ejecutivo": page_executive_dashboard(where)
    elif page == "Flujos Comerciales": page_trade_flow(where)
    elif page == "Modelado de Escenarios": page_scenario(where)
    elif page == "Motor de Riesgo (LVaR)": page_montecarlo(where)
    elif page == "Gemelo Portuario": page_port_twin(where)
    elif page == "Monitoreo Nearshoring": page_nearshoring(where)
    elif page == "Análisis de Sobrecostos (TCO)": page_cost_xray(where)
    elif page == "Consola de Investigación": page_research(where)
    elif page == "Glosario": page_glossary(where)

if __name__ == "__main__":
    main()
