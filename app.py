import streamlit as st
import numpy as np
import pandas as pd
import copy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64


# ---- Must install CoolProp: pip install CoolProp ----
from CoolProp.CoolProp import PropsSI

# ==============================================================
# PAGE CONFIG
# ==============================================================
st.set_page_config(
    page_title="HX Efficiency Analyzer",

    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# CUSTOM CSS
# ==============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .good { color: #16a34a; }
    .warn { color: #d97706; }
    .bad { color: #dc2626; }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .action-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    .action-box-good {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown label,
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================
# FLUID PROPERTY FUNCTIONS
# ==============================================================
def C_to_K(T_C):
    return T_C + 273.15

def bara_to_Pa(P_bara):
    return P_bara * 1e5

def Nm3h_to_kgs(flowrate_Nm3h, fluid):
    T_n, P_n = 273.15, 101325
    rho = PropsSI('D', 'T', T_n, 'P', P_n, fluid)
    return flowrate_Nm3h * rho / 3600.0

def get_specific_enthalpy(fluid, T_C, P_bara, phase='gas'):
    T_K = C_to_K(T_C)
    P_Pa = bara_to_Pa(P_bara)
    try:
        return PropsSI('H', 'T', T_K, 'P', P_Pa, fluid)
    except Exception:
        try:
            q = 0 if phase == 'liquid' else 1
            return PropsSI('H', 'P', P_Pa, 'Q', q, fluid)
        except Exception:
            return None

def get_saturation_temp(fluid, P_bara):
    try:
        return PropsSI('T', 'P', bara_to_Pa(P_bara), 'Q', 0.5, fluid) - 273.15
    except Exception:
        return None

# ==============================================================
# DESIGN SPECIFICATIONS
# ==============================================================
DESIGN_SPECS = {
    'AGMP': {
        'fluid': 'Air', 'total_flowrate_Nm3h': 27100,
        'vapor_in_Nm3h': 27100, 'vapor_out_Nm3h': 26932,
        'liquid_in_Nm3h': 0, 'liquid_out_Nm3h': 168,
        'T_in_C': 30.0, 'T_out_C': -172.7,
        'P_operating_bara': 5.891, 'pressure_drop_mbar': 164,
        'heat_duty_kcalh': 1798266, 'phase_change': True, 'stream_type': 'hot'
    },
    'AGHP': {
        'fluid': 'Air', 'total_flowrate_Nm3h': 14300,
        'vapor_in_Nm3h': 14300, 'vapor_out_Nm3h': 0,
        'liquid_in_Nm3h': 0, 'liquid_out_Nm3h': 14300,
        'T_in_C': 40.0, 'T_out_C': -170.0,
        'P_operating_bara': 62.680, 'pressure_drop_mbar': 173,
        'heat_duty_kcalh': 1670189, 'phase_change': True, 'stream_type': 'hot'
    },
    'AGT': {
        'fluid': 'Air', 'total_flowrate_Nm3h': 13500,
        'vapor_in_Nm3h': 13500, 'vapor_out_Nm3h': 13500,
        'liquid_in_Nm3h': 0, 'liquid_out_Nm3h': 0,
        'T_in_C': 40.0, 'T_out_C': -110.0,
        'P_operating_bara': 44.890, 'pressure_drop_mbar': 126,
        'heat_duty_kcalh': 747686, 'phase_change': True, 'stream_type': 'hot'
    },
    'GOX': {
        'fluid': 'Oxygen', 'total_flowrate_Nm3h': 11107,
        'vapor_in_Nm3h': 0, 'vapor_out_Nm3h': 11107,
        'liquid_in_Nm3h': 11107, 'liquid_out_Nm3h': 0,
        'T_in_C': -177.0, 'T_out_C': 28.3,
        'P_operating_bara': 31.352, 'pressure_drop_mbar': 198,
        'heat_duty_kcalh': 1466526, 'phase_change': True, 'stream_type': 'cold'
    },
    'GAN': {
        'fluid': 'Nitrogen', 'total_flowrate_Nm3h': 4000,
        'vapor_in_Nm3h': 4000, 'vapor_out_Nm3h': 4000,
        'liquid_in_Nm3h': 0, 'liquid_out_Nm3h': 0,
        'T_in_C': -175.5, 'T_out_C': 28.3,
        'P_operating_bara': 1.359, 'pressure_drop_mbar': 103,
        'heat_duty_kcalh': 255814, 'phase_change': False, 'stream_type': 'cold'
    },
    'WN2': {
        'fluid': 'Nitrogen', 'total_flowrate_Nm3h': 39085,
        'vapor_in_Nm3h': 39085, 'vapor_out_Nm3h': 39085,
        'liquid_in_Nm3h': 0, 'liquid_out_Nm3h': 0,
        'T_in_C': -175.5, 'T_out_C': 28.3,
        'P_operating_bara': 1.346, 'pressure_drop_mbar': 208,
        'heat_duty_kcalh': 2497700, 'phase_change': False, 'stream_type': 'cold'
    }
}

# ==============================================================
# COMPUTATION FUNCTIONS
# ==============================================================
def calculate_heat_duty(stream_name, spec):
    fluid = spec['fluid']
    m_dot = Nm3h_to_kgs(spec['total_flowrate_Nm3h'], fluid)
    total_flow = spec['total_flowrate_Nm3h']

    # Inlet enthalpy
    if spec['liquid_in_Nm3h'] > 0 and spec['vapor_in_Nm3h'] == 0:
        h_in = get_specific_enthalpy(fluid, spec['T_in_C'], spec['P_operating_bara'], 'liquid')
    elif spec['liquid_in_Nm3h'] > 0 and spec['vapor_in_Nm3h'] > 0:
        fv = spec['vapor_in_Nm3h'] / total_flow
        fl = spec['liquid_in_Nm3h'] / total_flow
        hv = get_specific_enthalpy(fluid, spec['T_in_C'], spec['P_operating_bara'], 'gas')
        hl = get_specific_enthalpy(fluid, spec['T_in_C'], spec['P_operating_bara'], 'liquid')
        h_in = fv * hv + fl * hl if hv and hl else None
    else:
        h_in = get_specific_enthalpy(fluid, spec['T_in_C'], spec['P_operating_bara'], 'gas')

    # Outlet enthalpy
    if spec['liquid_out_Nm3h'] > 0 and spec['vapor_out_Nm3h'] == 0:
        h_out = get_specific_enthalpy(fluid, spec['T_out_C'], spec['P_operating_bara'], 'liquid')
    elif spec['liquid_out_Nm3h'] > 0 and spec['vapor_out_Nm3h'] > 0:
        fv = spec['vapor_out_Nm3h'] / total_flow
        fl = spec['liquid_out_Nm3h'] / total_flow
        hv = get_specific_enthalpy(fluid, spec['T_out_C'], spec['P_operating_bara'], 'gas')
        hl = get_specific_enthalpy(fluid, spec['T_out_C'], spec['P_operating_bara'], 'liquid')
        h_out = fv * hv + fl * hl if hv and hl else None
    else:
        h_out = get_specific_enthalpy(fluid, spec['T_out_C'], spec['P_operating_bara'], 'gas')

    Q_W = m_dot * abs(h_out - h_in) if h_in and h_out else None
    Q_kcal = Q_W * 3600 / 4184 if Q_W else None

    return {'m_dot': m_dot, 'Q_W': Q_W, 'Q_kcalh': Q_kcal, 'h_in': h_in, 'h_out': h_out}


def compute_full_case(specs):
    results = {}
    for name, spec in specs.items():
        results[name] = calculate_heat_duty(name, spec)

    hot_T_max = max(specs[s]['T_in_C'] for s in specs if specs[s]['stream_type'] == 'hot')
    cold_T_min = min(specs[s]['T_in_C'] for s in specs if specs[s]['stream_type'] == 'cold')
    dT_max = hot_T_max - cold_T_min

    eps_per = {}
    for name, spec in specs.items():
        dT = abs(spec['T_out_C'] - spec['T_in_C'])
        eps_per[name] = dT / dT_max if dT_max > 0 else 0

    eps_overall = np.mean(list(eps_per.values()))
    Q_hot = sum(results[n]['Q_kcalh'] for n in specs if specs[n]['stream_type'] == 'hot' and results[n]['Q_kcalh'])
    Q_cold = sum(results[n]['Q_kcalh'] for n in specs if specs[n]['stream_type'] == 'cold' and results[n]['Q_kcalh'])

    return {'results': results, 'eps_per': eps_per, 'eps_overall': eps_overall,
            'Q_hot': Q_hot, 'Q_cold': Q_cold, 'dT_max': dT_max}


def run_root_cause(design_specs, plant_data, eps_design, eps_plant):
    root_causes = []
    for name in design_specs.keys():
        d_spec = design_specs[name]
        p_spec = plant_data[name]

        # Flow effect
        case_f = copy.deepcopy(design_specs)
        case_f[name]['total_flowrate_Nm3h'] = p_spec['total_flowrate_Nm3h']
        if d_spec['total_flowrate_Nm3h'] > 0:
            r = p_spec['total_flowrate_Nm3h'] / d_spec['total_flowrate_Nm3h']
            for k in ['vapor_in_Nm3h', 'vapor_out_Nm3h', 'liquid_in_Nm3h', 'liquid_out_Nm3h']:
                case_f[name][k] = d_spec[k] * r
        dev_f = (compute_full_case(case_f)['eps_overall'] - eps_design) / eps_design * 100

        # Tin effect
        case_t = copy.deepcopy(design_specs)
        case_t[name]['T_in_C'] = p_spec['T_in_C']
        dev_t = (compute_full_case(case_t)['eps_overall'] - eps_design) / eps_design * 100

        # Tout effect
        case_o = copy.deepcopy(design_specs)
        case_o[name]['T_out_C'] = p_spec['T_out_C']
        dev_o = (compute_full_case(case_o)['eps_overall'] - eps_design) / eps_design * 100

        # Combined
        case_c = copy.deepcopy(design_specs)
        case_c[name] = copy.deepcopy(p_spec)
        dev_c = (compute_full_case(case_c)['eps_overall'] - eps_design) / eps_design * 100

        root_causes.append({
            'Stream': name, 'Type': d_spec['stream_type'],
            'Flow Δ (%)': round((p_spec['total_flowrate_Nm3h'] - d_spec['total_flowrate_Nm3h']) / d_spec['total_flowrate_Nm3h'] * 100, 2),
            'Tin Δ (°C)': round(p_spec['T_in_C'] - d_spec['T_in_C'], 1),
            'Tout Δ (°C)': round(p_spec['T_out_C'] - d_spec['T_out_C'], 1),
            'ε impact: Flow (%)': round(dev_f, 4),
            'ε impact: Tin (%)': round(dev_t, 4),
            'ε impact: Tout (%)': round(dev_o, 4),
            'ε impact: Combined (%)': round(dev_c, 4),
        })
    return pd.DataFrame(root_causes)


def run_optimization(design_specs, plant_data, eps_design):
    from scipy.optimize import differential_evolution
    stream_names = list(design_specs.keys())

    def objective(x):
        case = copy.deepcopy(plant_data)
        for i, name in enumerate(stream_names):
            case[name]['T_in_C'] = plant_data[name]['T_in_C'] + x[i]
            ff = 1 + x[i + 6] / 100
            case[name]['total_flowrate_Nm3h'] = plant_data[name]['total_flowrate_Nm3h'] * ff
            for k in ['vapor_in_Nm3h', 'vapor_out_Nm3h', 'liquid_in_Nm3h', 'liquid_out_Nm3h']:
                case[name][k] = plant_data[name][k] * ff

        hot_T = max(case[s]['T_in_C'] for s in case if case[s]['stream_type'] == 'hot')
        cold_T = min(case[s]['T_in_C'] for s in case if case[s]['stream_type'] == 'cold')
        dT = hot_T - cold_T
        if dT <= 0: return 1e6
        eps_vals = [abs(case[n]['T_out_C'] - case[n]['T_in_C']) / dT for n in stream_names]
        eps_o = np.mean(eps_vals)
        penalty = 5000 * (eps_o - eps_design) ** 2
        cost = sum(x[i]**2 for i in range(6)) + sum(x[i+6]**2 for i in range(6))
        return cost + penalty

    bounds = ([(-15, 15)] * 6) + ([(-15, 15)] * 6)
    result = differential_evolution(objective, bounds, seed=42, maxiter=500, tol=1e-12, popsize=30)
    x = result.x

    recs = []
    for i, name in enumerate(stream_names):
        tin_adj = x[i]
        flow_adj = x[i + 6]
        recs.append({
            'Stream': name,
            'Type': design_specs[name]['stream_type'],
            'Tin current (°C)': plant_data[name]['T_in_C'],
            'Tin adjust (°C)': round(tin_adj, 2),
            'Tin target (°C)': round(plant_data[name]['T_in_C'] + tin_adj, 2),
            'Tin design (°C)': design_specs[name]['T_in_C'],
            'Flow current': plant_data[name]['total_flowrate_Nm3h'],
            'Flow adjust (%)': round(flow_adj, 2),
            'Flow target': round(plant_data[name]['total_flowrate_Nm3h'] * (1 + flow_adj/100), 0),
            'Flow design': design_specs[name]['total_flowrate_Nm3h'],
            'Feasible': '✅' if abs(tin_adj) < 10 and abs(flow_adj) < 8 else '⚠️'
        })
    return pd.DataFrame(recs), result.success


# ==============================================================
# COMPUTE DESIGN BASELINE (cached)
# ==============================================================
@st.cache_data
def get_design_baseline():
    case = compute_full_case(DESIGN_SPECS)
    return case['eps_overall'], case


design_eps, design_case = get_design_baseline()


# ==============================================================
# SIDEBAR — Plant Data Input
# ==============================================================
st.sidebar.markdown("## 🏭 Plant Operating Data")
st.sidebar.markdown("Enter current plant values below.")
st.sidebar.markdown("---")

plant_data = copy.deepcopy(DESIGN_SPECS)

stream_colors = {
    'AGMP': '🔴', 'AGHP': '🟠', 'AGT': '🟡',
    'GOX': '🔵', 'GAN': '🟢', 'WN2': '🟣'
}

for name in DESIGN_SPECS.keys():
    spec = DESIGN_SPECS[name]
    st.sidebar.markdown(f"### {stream_colors[name]} {name} ({spec['fluid']}, {spec['stream_type']})")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        flow = st.number_input(
            f"Flow (Nm³/h)", value=float(spec['total_flowrate_Nm3h']),
            step=100.0, key=f"flow_{name}", format="%.0f"
        )
        t_in = st.number_input(
            f"T_in (°C)", value=float(spec['T_in_C']),
            step=0.5, key=f"tin_{name}", format="%.1f"
        )
    with col2:
        t_out = st.number_input(
            f"T_out (°C)", value=float(spec['T_out_C']),
            step=0.5, key=f"tout_{name}", format="%.1f"
        )
        dp = st.number_input(
            f"ΔP (mbar)", value=float(spec['pressure_drop_mbar']),
            step=5.0, key=f"dp_{name}", format="%.0f"
        )

    plant_data[name]['total_flowrate_Nm3h'] = flow
    plant_data[name]['T_in_C'] = t_in
    plant_data[name]['T_out_C'] = t_out
    plant_data[name]['pressure_drop_mbar'] = dp

    # Scale vapor/liquid flows proportionally
    if spec['total_flowrate_Nm3h'] > 0:
        ratio = flow / spec['total_flowrate_Nm3h']
        for k in ['vapor_in_Nm3h', 'vapor_out_Nm3h', 'liquid_in_Nm3h', 'liquid_out_Nm3h']:
            plant_data[name][k] = spec[k] * ratio

    st.sidebar.markdown("---")


# ==============================================================
# MAIN CONTENT
# ==============================================================
st.markdown('<div class="main-header">🔥 Heat Exchanger Efficiency Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Spec Sheet N° 4351-1 · Counter-Flow Plate-Fin · JSPL India · Main HX (ASU)</div>', unsafe_allow_html=True)

# --- Compute plant case ---
plant_case = compute_full_case(plant_data)
eps_plant = plant_case['eps_overall']
efficiency = (eps_plant / design_eps) * 100
deviation = efficiency - 100

# ==============================================================
# TAB LAYOUT
# ==============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Efficiency & Effectiveness",
    "🔍 Root Cause Analysis",
    "🎯 Suggested Adjustments",
    "📋 Design Reference"
])


# ==============================================================
# TAB 1: EFFICIENCY & EFFECTIVENESS
# ==============================================================
with tab1:
    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)

    dev_class = "good" if abs(deviation) < 1 else "warn" if abs(deviation) < 3 else "bad"
    eff_class = "good" if efficiency > 99 else "warn" if efficiency > 97 else "bad"

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Overall Effectiveness (ε)</div>
            <div class="metric-value">{eps_plant:.4f}</div>
            <div style="color:#64748b;font-size:0.85rem;">Design: {design_eps:.4f}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Efficiency</div>
            <div class="metric-value {eff_class}">{efficiency:.2f}%</div>
            <div style="color:#64748b;font-size:0.85rem;">Design = 100%</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Deviation</div>
            <div class="metric-value {dev_class}">{deviation:+.2f}%</div>
            <div style="color:#64748b;font-size:0.85rem;">Target: 0%</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        bal_err = abs(plant_case['Q_hot'] - plant_case['Q_cold']) / max(plant_case['Q_hot'], plant_case['Q_cold']) * 100
        bal_class = "good" if bal_err < 3 else "warn" if bal_err < 8 else "bad"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Energy Balance Error</div>
            <div class="metric-value {bal_class}">{bal_err:.1f}%</div>
            <div style="color:#64748b;font-size:0.85rem;">Q_hot vs Q_cold</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Per-stream comparison table
    st.markdown('<div class="section-title">Per-Stream Comparison</div>', unsafe_allow_html=True)

    rows = []
    for name in DESIGN_SPECS.keys():
        d = DESIGN_SPECS[name]
        p = plant_data[name]
        pr = plant_case['results'][name]
        q_p = pr['Q_kcalh']
        q_d = d['heat_duty_kcalh']
        duty_dev = ((q_p - q_d) / q_d * 100) if q_p else None

        rows.append({
            'Stream': name,
            'Type': d['stream_type'].upper(),
            'Flow Plant': f"{p['total_flowrate_Nm3h']:,.0f}",
            'Flow Design': f"{d['total_flowrate_Nm3h']:,.0f}",
            'Tin Plant (°C)': p['T_in_C'],
            'Tin Design (°C)': d['T_in_C'],
            'Tout Plant (°C)': p['T_out_C'],
            'Tout Design (°C)': d['T_out_C'],
            'Q Plant (kcal/h)': f"{q_p:,.0f}" if q_p else "N/A",
            'Q Design (kcal/h)': f"{q_d:,.0f}",
            'Duty Dev (%)': f"{duty_dev:+.2f}" if duty_dev else "N/A",
            'ε Stream': f"{plant_case['eps_per'][name]:.4f}"
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Charts
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        streams = list(DESIGN_SPECS.keys())
        q_d = [DESIGN_SPECS[s]['heat_duty_kcalh']/1e6 for s in streams]
        q_p = [plant_case['results'][s]['Q_kcalh']/1e6 if plant_case['results'][s]['Q_kcalh'] else 0 for s in streams]
        fig.add_trace(go.Bar(name='Design', x=streams, y=q_d, marker_color='#3b82f6'))
        fig.add_trace(go.Bar(name='Plant', x=streams, y=q_p, marker_color='#f97316'))
        fig.update_layout(title='Heat Duty Comparison (×10⁶ kcal/h)', barmode='group',
                         height=380, template='plotly_white',
                         font=dict(family='DM Sans'), margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        colors_map = {'AGMP': '#ef4444', 'AGHP': '#f97316', 'AGT': '#eab308',
                      'GOX': '#3b82f6', 'GAN': '#22c55e', 'WN2': '#a855f7'}
        for s in streams:
            fig2.add_trace(go.Scatter(
                x=['Inlet', 'Outlet'],
                y=[plant_data[s]['T_in_C'], plant_data[s]['T_out_C']],
                mode='lines+markers', name=s,
                line=dict(width=3, color=colors_map[s]),
                marker=dict(size=10)
            ))
        fig2.update_layout(title='Temperature Profiles', yaxis_title='Temperature (°C)',
                          height=380, template='plotly_white',
                          font=dict(family='DM Sans'), margin=dict(t=40, b=40))
        st.plotly_chart(fig2, use_container_width=True)


# ==============================================================
# TAB 2: ROOT CAUSE ANALYSIS
# ==============================================================
with tab2:
    st.markdown('<div class="section-title">Root Cause Decomposition</div>', unsafe_allow_html=True)

    with st.spinner("Running sensitivity analysis..."):
        df_rc = run_root_cause(DESIGN_SPECS, plant_data, design_eps, eps_plant)

    # Summary metrics
    total_flow_eff = df_rc['ε impact: Flow (%)'].sum()
    total_tin_eff = df_rc['ε impact: Tin (%)'].sum()
    total_tout_eff = df_rc['ε impact: Tout (%)'].sum()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Flowrate Effect", f"{total_flow_eff:+.4f}%")
    with c2:
        st.metric("Inlet Temp Effect", f"{total_tin_eff:+.4f}%")
    with c3:
        st.metric("Outlet Temp Effect", f"{total_tout_eff:+.4f}%")

    # Dominant factor
    factors = {'Flowrate': abs(total_flow_eff), 'Inlet Temperature': abs(total_tin_eff),
               'Outlet Temperature': abs(total_tout_eff)}
    dominant = max(factors, key=factors.get)
    st.info(f"🎯 **Dominant Factor:** {dominant} changes")

    # Table
    st.dataframe(df_rc.sort_values('ε impact: Combined (%)', key=abs, ascending=False),
                 use_container_width=True, hide_index=True)

    # Ranked chart
    df_sorted = df_rc.sort_values('ε impact: Combined (%)', key=abs, ascending=True)
    fig_rc = go.Figure()
    fig_rc.add_trace(go.Bar(
        y=df_sorted['Stream'],
        x=df_sorted['ε impact: Flow (%)'],
        name='Flow', orientation='h', marker_color='#3b82f6'
    ))
    fig_rc.add_trace(go.Bar(
        y=df_sorted['Stream'],
        x=df_sorted['ε impact: Tin (%)'],
        name='Inlet Temp', orientation='h', marker_color='#f97316'
    ))
    fig_rc.add_trace(go.Bar(
        y=df_sorted['Stream'],
        x=df_sorted['ε impact: Tout (%)'],
        name='Outlet Temp', orientation='h', marker_color='#22c55e'
    ))
    fig_rc.update_layout(barmode='group', title='ε Impact Decomposition by Stream',
                        xaxis_title='ε Deviation (%)', height=400, template='plotly_white',
                        font=dict(family='DM Sans'), margin=dict(t=40, b=40, l=80))
    st.plotly_chart(fig_rc, use_container_width=True)


# ==============================================================
# TAB 3: SUGGESTED ADJUSTMENTS
# ==============================================================
with tab3:
    st.markdown('<div class="section-title">Recommended Inlet Adjustments</div>', unsafe_allow_html=True)
    st.markdown("These are the **minimum changes to controllable inputs** (inlet temps & flowrates) to reach design efficiency.")

    if abs(deviation) < 0.05:
        st.markdown('<div class="action-box-good">✅ <b>Plant is already at design efficiency!</b> No adjustments needed.</div>',
                   unsafe_allow_html=True)
    else:
        with st.spinner("Running optimizer (this may take 10-20 seconds)..."):
            df_opt, converged = run_optimization(DESIGN_SPECS, plant_data, design_eps)

        if converged:
            st.success("Optimizer converged successfully.")
        else:
            st.warning("Optimizer did not fully converge. Results are approximate.")

        st.dataframe(df_opt, use_container_width=True, hide_index=True)

        # Action summary
        st.markdown('<div class="section-title">Action Summary</div>', unsafe_allow_html=True)

        for _, row in df_opt.iterrows():
            tin_adj = row['Tin adjust (°C)']
            flow_adj = row['Flow adjust (%)']
            name = row['Stream']

            if abs(tin_adj) < 0.5 and abs(flow_adj) < 0.5:
                st.markdown(f'<div class="action-box-good">✅ <b>{name}</b> — No significant adjustment needed</div>',
                           unsafe_allow_html=True)
            else:
                actions = []
                if abs(tin_adj) > 0.5:
                    d = "Increase" if tin_adj > 0 else "Decrease"
                    actions.append(f"Inlet Temp: <b>{d} by {abs(tin_adj):.1f}°C</b> "
                                  f"({row['Tin current (°C)']}°C → {row['Tin target (°C)']}°C)")
                if abs(flow_adj) > 0.5:
                    d = "Increase" if flow_adj > 0 else "Decrease"
                    actions.append(f"Flowrate: <b>{d} by {abs(flow_adj):.1f}%</b> "
                                  f"({row['Flow current']:,.0f} → {row['Flow target']:,.0f} Nm³/h)")

                box_class = "action-box"
                st.markdown(f'<div class="{box_class}">🔧 <b>{name}</b> {row["Feasible"]}<br>'
                           + '<br>'.join(actions) + '</div>', unsafe_allow_html=True)

        # Visualization
        fig_opt = make_subplots(rows=1, cols=2,
                                subplot_titles=('Inlet Temp Adjustments', 'Flowrate Adjustments'))
        fig_opt.add_trace(go.Bar(
            x=df_opt['Stream'], y=df_opt['Tin adjust (°C)'],
            marker_color=['#ef4444' if v < 0 else '#22c55e' for v in df_opt['Tin adjust (°C)']],
            text=[f"{v:+.1f}°C" for v in df_opt['Tin adjust (°C)']],
            textposition='outside'
        ), row=1, col=1)
        fig_opt.add_trace(go.Bar(
            x=df_opt['Stream'], y=df_opt['Flow adjust (%)'],
            marker_color=['#ef4444' if v < 0 else '#22c55e' for v in df_opt['Flow adjust (%)']],
            text=[f"{v:+.1f}%" for v in df_opt['Flow adjust (%)']],
            textposition='outside'
        ), row=1, col=2)
        fig_opt.update_layout(height=400, showlegend=False, template='plotly_white',
                             font=dict(family='DM Sans'), margin=dict(t=50, b=40))
        fig_opt.update_yaxes(title_text="°C", row=1, col=1)
        fig_opt.update_yaxes(title_text="%", row=1, col=2)
        st.plotly_chart(fig_opt, use_container_width=True)


# ==============================================================
# TAB 4: DESIGN REFERENCE
# ==============================================================
with tab4:
    st.markdown('<div class="section-title">Design Specifications (Datasheet N° 4351-1)</div>', unsafe_allow_html=True)

    ref_rows = []
    for name, spec in DESIGN_SPECS.items():
        ref_rows.append({
            'Stream': name,
            'Fluid': spec['fluid'],
            'Type': spec['stream_type'].upper(),
            'Flow (Nm³/h)': f"{spec['total_flowrate_Nm3h']:,}",
            'T_in (°C)': spec['T_in_C'],
            'T_out (°C)': spec['T_out_C'],
            'P (bara)': spec['P_operating_bara'],
            'ΔP (mbar)': spec['pressure_drop_mbar'],
            'Q (kcal/h)': f"{spec['heat_duty_kcalh']:,}",
            'Phase Change': '✅' if spec['phase_change'] else '—'
        })

    df_design_ref = pd.DataFrame(ref_rows)
    st.dataframe(df_design_ref, use_container_width=True, hide_index=True)

    # Re-calculate comparison dataframe for export
    comparison = []
    for name in DESIGN_SPECS.keys():
        d = DESIGN_SPECS[name]
        p = plant_data[name]
        pr = plant_case['results'][name]
        q_p = pr['Q_kcalh']
        q_d = d['heat_duty_kcalh']
        duty_dev = ((q_p - q_d) / q_d * 100) if q_p else None
        
        comparison.append({
            'Stream': name,
            'Type': d['stream_type'],
            'Flow_design (Nm3/h)': d['total_flowrate_Nm3h'],
            'Flow_plant (Nm3/h)': p['total_flowrate_Nm3h'],
            'Flow_dev (%)': round((p['total_flowrate_Nm3h'] - d['total_flowrate_Nm3h']) / d['total_flowrate_Nm3h'] * 100, 2),
            'Tin_design (°C)': d['T_in_C'],
            'Tin_plant (°C)': p['T_in_C'],
            'Tin_dev (°C)': round(p['T_in_C'] - d['T_in_C'], 1),
            'Tout_design (°C)': d['T_out_C'],
            'Tout_plant (°C)': p['T_out_C'],
            'Tout_dev (°C)': round(p['T_out_C'] - d['T_out_C'], 1),
            'Q_design (kcal/h)': q_d,
            'Q_plant (kcal/h)': round(q_p, 0) if q_p else None,
            'Duty_dev (%)': round(duty_dev, 2) if duty_dev else None,
            'Duty_ratio': round(q_p / q_d, 4) if q_p and q_d else None
        })
    df_comparison = pd.DataFrame(comparison)

    st.markdown('<div class="section-title">Export Data</div>', unsafe_allow_html=True)

    def to_excel(df_design, df_comparison_data, df_rc_data, df_opt_data):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_design.to_excel(writer, sheet_name='Design_Specs', index=False)
            df_comparison_data.to_excel(writer, sheet_name='Plant_vs_Design', index=False)
            df_rc_data.to_excel(writer, sheet_name='Root_Cause_Analysis', index=False)
            df_opt_data.to_excel(writer, sheet_name='Optimized_Adjustments', index=False)
        processed_data = output.getvalue()
        return processed_data

    # Run root cause and optimization again to get the latest dataframes for export
    df_rc_export = run_root_cause(DESIGN_SPECS, plant_data, design_eps, eps_plant)
    df_opt_export, _ = run_optimization(DESIGN_SPECS, plant_data, design_eps)

    excel_data = to_excel(df_design_ref, df_comparison, df_rc_export, df_opt_export)
    st.download_button(
        label="📥 Download Full Report (Excel)",
        data=excel_data,
        file_name="HX_Efficiency_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown(f"""
    **HX Details:**
    - Construction Code: ASME with "U" Stamp
    - Type: Counter-flow plate-fin, 2 cores
    - Dimensions: 1065 × 1084 × 4240 mm
    - Total layers/core: 107 (+2 dummy)
    - Total heat transfer area: 9,672 m³
    """)

    st.markdown(f"""
    **HX Details:**
    - Construction Code: ASME with "U" Stamp
    - Type: Counter-flow plate-fin, 2 cores
    - Dimensions: 1065 × 1084 × 4240 mm
    - Total layers/core: 107 (+2 dummy)
    - Total heat transfer area: 9,672 m²
    """)
