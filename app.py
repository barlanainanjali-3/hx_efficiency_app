import streamlit as st
import numpy as np
import pandas as pd
import copy
from CoolProp.CoolProp import PropsSI

# ============================================================
# Streamlit App: Heat Exchanger Efficiency Model
# ============================================================

st.set_page_config(
    page_title="HX Efficiency Model",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Plate-Fin Heat Exchanger Efficiency Model")
st.markdown("### Multi-Stream Counter-Flow — JSPL India (ASU)")
st.write("Adjust the parameters below to analyze real-time plant performance against design specifications.")

# ============================================================
# Static Design Specifications (from Cell 2)
# ============================================================

# Fluid mapping for CoolProp
# AGMP/AGHP/AGT = Air, GOX = Oxygen, GAN/WN2 = Nitrogen
FLUID_MAP = {
    'AGMP': 'Air',
    'AGHP': 'Air',
    'AGT':  'Air',
    'GOX':  'Oxygen',
    'GAN':  'Nitrogen',
    'WN2':  'Nitrogen'
}

# Design specifications dictionary
# All flowrates in Nm3/h, temperatures in °C, pressures in bar(a)
# pressure_drop in mbar, heat_duty_design in kcal/h
design_specs = {
    'AGMP': {
        'fluid': 'Air',
        'total_flowrate_Nm3h': 27100,
        'vapor_in_Nm3h': 27100,
        'vapor_out_Nm3h': 26932,
        'liquid_in_Nm3h': 0,
        'liquid_out_Nm3h': 168,
        'T_in_C': 30.0,
        'T_out_C': -172.7,
        'P_operating_bara': 5.891,
        'pressure_drop_mbar': 164,
        'heat_duty_kcalh': 1798266,
        'phase_change': True,  # small liquid out
        'stream_type': 'hot'   # hot stream being cooled
    },
    'AGHP': {
        'fluid': 'Air',
        'total_flowrate_Nm3h': 14300,
        'vapor_in_Nm3h': 14300,
        'vapor_out_Nm3h': 0,
        'liquid_in_Nm3h': 0,
        'liquid_out_Nm3h': 14300,
        'T_in_C': 40.0,
        'T_out_C': -170.0,
        'P_operating_bara': 62.680,
        'pressure_drop_mbar': 173,
        'heat_duty_kcalh': 1670189,
        'phase_change': True,  # fully condensed
        'stream_type': 'hot'
    },
    'AGT': {
        'fluid': 'Air',
        'total_flowrate_Nm3h': 13500,
        'vapor_in_Nm3h': 13500,
        'vapor_out_Nm3h': 13500,
        'liquid_in_Nm3h': 0,
        'liquid_out_Nm3h': 0,
        'T_in_C': 40.0,
        'T_out_C': -110.0,
        'P_operating_bara': 44.890,
        'pressure_drop_mbar': 126,
        'heat_duty_kcalh': 747686,
        'phase_change': True,  # user confirmed phase change at inlet
        'stream_type': 'hot'
    },
    'GOX': {
        'fluid': 'Oxygen',
        'total_flowrate_Nm3h': 11107,
        'vapor_in_Nm3h': 0,
        'vapor_out_Nm3h': 11107,
        'liquid_in_Nm3h': 11107,
        'liquid_out_Nm3h': 0,
        'T_in_C': -177.0,
        'T_out_C': 28.3,
        'P_operating_bara': 31.352,
        'pressure_drop_mbar': 198,
        'heat_duty_kcalh': 1466526,
        'phase_change': True,  # liquid in -> vapor out (vaporization)
        'stream_type': 'cold'  # cold stream being heated
    },
    'GAN': {
        'fluid': 'Nitrogen',
        'total_flowrate_Nm3h': 4000,
        'vapor_in_Nm3h': 4000,
        'vapor_out_Nm3h': 4000,
        'liquid_in_Nm3h': 0,
        'liquid_out_Nm3h': 0,
        'T_in_C': -175.5,
        'T_out_C': 28.3,
        'P_operating_bara': 1.359,
        'pressure_drop_mbar': 103,
        'heat_duty_kcalh': 255814,
        'phase_change': False,
        'stream_type': 'cold'
    },
    'WN2': {
        'fluid': 'Nitrogen',
        'total_flowrate_Nm3h': 39085,
        'vapor_in_Nm3h': 39085,
        'vapor_out_Nm3h': 39085,
        'liquid_in_Nm3h': 0,
        'liquid_out_Nm3h': 0,
        'T_in_C': -175.5,
        'T_out_C': 28.3,
        'P_operating_bara': 1.346,
        'pressure_drop_mbar': 208,
        'heat_duty_kcalh': 2497700,
        'phase_change': False,
        'stream_type': 'cold'
    }
}

# ============================================================
# Core Functions Extracted for Utility/Streamlit App (from previous cells)
# (These functions will be copied from b2ec2216)
# ============================================================

def C_to_K(T_C):
    """Convert Celsius to Kelvin."""
    return T_C + 273.15

def bara_to_Pa(P_bara):
    """Convert bar(absolute) to Pascal."""
    return P_bara * 1e5

def Nm3h_to_kgs(flowrate_Nm3h, fluid):
    """
    Convert Nm³/h to kg/s.
    Normal conditions: 0°C (273.15K), 1.01325 bar (101325 Pa).
    """
    T_normal = 273.15  # K
    P_normal = 101325  # Pa
    # Density at normal conditions
    rho_normal = PropsSI('D', 'T', T_normal, 'P', P_normal, fluid)  # kg/m³
    mass_flow_kgh = flowrate_Nm3h * rho_normal  # kg/h
    mass_flow_kgs = mass_flow_kgh / 3600.0  # kg/s
    return mass_flow_kgs

def get_specific_enthalpy(fluid, T_C, P_bara, phase='gas'):
    """
    Get specific enthalpy (J/kg) using CoolProp.
    For two-phase or near-saturation, we use T and P directly.
    If CoolProp fails (e.g., two-phase), we try phase-specific lookup.
    """
    T_K = C_to_K(T_C)
    P_Pa = bara_to_Pa(P_bara)

    try:
        h = PropsSI('H', 'T', T_K, 'P', P_Pa, fluid)
        return h
    except Exception:
        # If in two-phase region, get saturated liquid or vapor enthalpy
        try:
            if phase == 'liquid':
                h = PropsSI('H', 'P', P_Pa, 'Q', 0, fluid)  # saturated liquid
            else:
                h = PropsSI('H', 'P', P_Pa, 'Q', 1, fluid)  # saturated vapor
            return h
        except Exception as e2:
            # print(f"  ⚠️  CoolProp fallback failed for {fluid} at T={T_C}°C, P={P_bara} bara: {e2}")
            return None

# From Cell 4: Heat Duty Calculation per Stream (Enthalpy-Based)
def calculate_heat_duty_enthalpy(stream_name, spec):
    """
    Calculate heat duty using enthalpy difference method.
    Q = m_dot * |h_out - h_in|  (in Watts)

    For streams with phase change:
    - If fluid enters as liquid and exits as vapor (or vice versa),
      we compute enthalpy at inlet and outlet conditions.
    - For partial phase change (e.g., AGMP: mostly vapor out + small liquid out),
      we compute weighted enthalpy at outlet.

    Returns: dict with Q_watts, Q_kcalh, h_in, h_out, m_dot, details
    """
    fluid = spec['fluid']
    T_in = spec['T_in_C']
    T_out = spec['T_out_C']
    P_bara = spec['P_operating_bara']
    total_flow = spec['total_flowrate_Nm3h']

    # Mass flow rate
    m_dot = Nm3h_to_kgs(total_flow, fluid)

    details = []

    # --- Determine inlet enthalpy ---
    vap_in = spec['vapor_in_Nm3h']
    liq_in = spec['liquid_in_Nm3h']

    if liq_in > 0 and vap_in == 0:
        # Pure liquid inlet (e.g., GOX)
        h_in = get_specific_enthalpy(fluid, T_in, P_bara, phase='liquid')
        details.append(f"Inlet: pure liquid at {T_in}°C")
    elif liq_in > 0 and vap_in > 0:
        # Mixed inlet — weighted average
        frac_vap = vap_in / total_flow
        frac_liq = liq_in / total_flow
        h_vap = get_specific_enthalpy(fluid, T_in, P_bara, phase='gas')
        h_liq = get_specific_enthalpy(fluid, T_in, P_bara, phase='liquid')
        h_in = frac_vap * h_vap + frac_liq * h_liq
        details.append(f"Inlet: mixed (vap={frac_vap:.2f}, liq={frac_liq:.2f})")
    else:
        # Pure vapor inlet
        h_in = get_specific_enthalpy(fluid, T_in, P_bara, phase='gas')
        details.append(f"Inlet: pure vapor at {T_in}°C")

    # --- Determine outlet enthalpy ---
    vap_out = spec['vapor_out_Nm3h']
    liq_out = spec['liquid_out_Nm3h']

    if liq_out > 0 and vap_out == 0:
        # Pure liquid outlet (e.g., AGHP fully condensed)
        h_out = get_specific_enthalpy(fluid, T_out, P_bara, phase='liquid')
        details.append(f"Outlet: pure liquid at {T_out}°C")
    elif liq_out > 0 and vap_out > 0:
        # Mixed outlet (e.g., AGMP: small liquid fraction)
        frac_vap = vap_out / total_flow
        frac_liq = liq_out / total_flow
        h_vap = get_specific_enthalpy(fluid, T_out, P_bara, phase='gas')
        h_liq = get_specific_enthalpy(fluid, T_out, P_bara, phase='liquid')
        h_out = frac_vap * h_vap + frac_liq * h_liq
        details.append(f"Outlet: mixed (vap={frac_vap:.2f}, liq={frac_liq:.2f})")
    else:
        # Pure vapor outlet
        h_out = get_specific_enthalpy(fluid, T_out, P_bara, phase='gas')
        details.append(f"Outlet: pure vapor at {T_out}°C")

    # Heat duty
    if h_in is not None and h_out is not None:
        Q_watts = m_dot * abs(h_out - h_in)
        Q_kcalh = Q_watts * 3600 / 4184  # W to kcal/h
    else:
        Q_watts = None
        Q_kcalh = None
        details.append("❌ Could not compute enthalpy")

    return {
        'stream': stream_name,
        'm_dot_kgs': m_dot,
        'h_in_Jkg': h_in,
        'h_out_Jkg': h_out,
        'Q_watts': Q_watts,
        'Q_kcalh': Q_kcalh,
        'Q_design_kcalh': spec['heat_duty_kcalh'],
        'details': details
    }

# From Cell 7: Helper function for efficiency computation (originally inside run_simulation)
def compute_efficiency_for_case(case_specs):
    """
    Given a modified set of specs, compute heat duties and effectiveness.
    Returns dict with per-stream Q, overall eps, and deviations.
    """
    results = {}
    for name, spec in case_specs.items():
        results[name] = calculate_heat_duty_enthalpy(name, spec)

    # Temperature effectiveness
    hot_T_in_max = max(s['T_in_C'] for s in case_specs.values() if s['stream_type'] == 'hot')
    cold_T_in_min = min(s['T_in_C'] for s in case_specs.values() if s['stream_type'] == 'cold')
    delta_T_max = hot_T_in_max - cold_T_in_min

    eps_per_stream = {}
    for name, spec in case_specs.items():
        dT = abs(spec['T_out_C'] - spec['T_in_C'])
        eps_per_stream[name] = dT / delta_T_max if delta_T_max > 0 else 0

    eps_overall = np.mean(list(eps_per_stream.values()))

    Q_hot = sum(results[n]['Q_kcalh'] for n in case_specs
                if case_specs[n]['stream_type'] == 'hot' and results[n]['Q_kcalh'])
    Q_cold = sum(results[n]['Q_kcalh'] for n in case_specs
                 if case_specs[n]['stream_type'] == 'cold' and results[n]['Q_kcalh'])

    return {
        'results': results,
        'eps_per_stream': eps_per_stream,
        'eps_overall': eps_overall,
        'Q_hot_total': Q_hot,
        'Q_cold_total': Q_cold
    }


# From Cell 8: Analyze plant data against design
def analyze_plant_vs_design(plant_data, design_specs, design_baseline):
    """
    Compare plant operating data against design baseline.
    Returns detailed comparison DataFrame.
    """
    # Compute plant heat duties
    plant_case = compute_efficiency_for_case(plant_data)

    comparison = []
    for name in design_specs.keys():
        d = design_baseline['per_stream'][name]
        p_result = plant_case['results'][name]
        p_spec = plant_data[name]

        # Flow deviation
        flow_dev = ((p_spec['total_flowrate_Nm3h'] - d['flowrate_Nm3h']) / d['flowrate_Nm3h'] * 100)

        # Temperature deviations
        T_in_dev = p_spec['T_in_C'] - d['T_in_C']
        T_out_dev = p_spec['T_out_C'] - d['T_out_C']

        # Duty deviation
        q_plant = p_result['Q_kcalh']
        q_design = d['Q_design_kcalh']
        duty_dev = ((q_plant - q_design) / q_design * 100) if q_plant else None

        # Duty ratio (plant actual / design)
        duty_ratio = q_plant / q_design if q_plant else None

        comparison.append({
            'Stream': name,
            'Type': design_specs[name]['stream_type'],
            'Flow_design': d['flowrate_Nm3h'],
            'Flow_plant': p_spec['total_flowrate_Nm3h'],
            'Flow_dev (%)': round(flow_dev, 2),
            'Tin_design': d['T_in_C'],
            'Tin_plant': p_spec['T_in_C'],
            'Tin_dev (°C)': T_in_dev,
            'Tout_design': d['T_out_C'],
            'Tout_plant': p_spec['T_out_C'],
            'Tout_dev (°C)': T_out_dev,
            'Q_design (kcal/h)': q_design,
            'Q_plant (kcal/h)': round(q_plant, 0) if q_plant else None,
            'Duty_dev (%)': round(duty_dev, 2) if duty_dev else None,
            'Duty_ratio': round(duty_ratio, 4) if duty_ratio else None
        })

    df_comp = pd.DataFrame(comparison)

    # Overall
    eps_plant = plant_case['eps_overall']
    eps_design = design_baseline['eps_overall']
    eps_dev = ((eps_plant - eps_design) / eps_design) * 100

    # Overall efficiency percentage (design = 100%)
    overall_efficiency = (eps_plant / eps_design) * 100

    return df_comp, eps_plant, overall_efficiency


# ============================================================
# Streamlit UI for user inputs
# ============================================================

st.header("1. Enter Current Plant Operating Data")
st.markdown("Adjust the sliders or type in the values for each stream's flowrate and temperatures.")

plant_data = copy.deepcopy(design_specs)

# Create tabs for Hot and Cold streams
hot_streams = [s for s, spec in design_specs.items() if spec['stream_type'] == 'hot']
cold_streams = [s for s, spec in design_specs.items() if spec['stream_type'] == 'cold']

hot_tab, cold_tab = st.tabs(["Hot Streams", "Cold Streams"])

# Hot Streams input
with hot_tab:
    st.subheader("Hot Streams (Cooling)")
    for stream_name in hot_streams:
        spec = design_specs[stream_name]
        with st.expander(f"**{stream_name}** (Design: {spec['T_in_C']}°C in, {spec['T_out_C']}°C out, {spec['total_flowrate_Nm3h']} Nm³/h)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                plant_data[stream_name]['total_flowrate_Nm3h'] = st.number_input(
                    f"Flowrate (Nm³/h) for {stream_name}",
                    min_value=0.0,
                    max_value=float(spec['total_flowrate_Nm3h'] * 1.5),
                    value=float(plant_data[stream_name]['total_flowrate_Nm3h']),
                    step=10.0,
                    format="%.1f",
                    key=f"flow_{stream_name}"
                )
            with col2:
                plant_data[stream_name]['T_in_C'] = st.number_input(
                    f"Inlet Temp (°C) for {stream_name}",
                    min_value=-200.0,
                    max_value=100.0,
                    value=float(plant_data[stream_name]['T_in_C']),
                    step=0.1,
                    format="%.1f",
                    key=f"Tin_{stream_name}"
                )
            with col3:
                plant_data[stream_name]['T_out_C'] = st.number_input(
                    f"Outlet Temp (°C) for {stream_name}",
                    min_value=-200.0,
                    max_value=100.0,
                    value=float(plant_data[stream_name]['T_out_C']),
                    step=0.1,
                    format="%.1f",
                    key=f"Tout_{stream_name}"
                )
            # Update vapor/liquid flowrates proportionally for calculation
            flow_ratio = plant_data[stream_name]['total_flowrate_Nm3h'] / spec['total_flowrate_Nm3h'] if spec['total_flowrate_Nm3h'] != 0 else 0
            for key_suffix in ['vapor_in_Nm3h', 'vapor_out_Nm3h', 'liquid_in_Nm3h', 'liquid_out_Nm3h']:
                original_key = key_suffix #f'{key_suffix}_{stream_name}' # this logic is not needed, original key is enough
                if original_key in spec:
                    plant_data[stream_name][original_key] = spec[original_key] * flow_ratio

# Cold Streams input
with cold_tab:
    st.subheader("Cold Streams (Heating)")
    for stream_name in cold_streams:
        spec = design_specs[stream_name]
        with st.expander(f"**{stream_name}** (Design: {spec['T_in_C']}°C in, {spec['T_out_C']}°C out, {spec['total_flowrate_Nm3h']} Nm³/h)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                plant_data[stream_name]['total_flowrate_Nm3h'] = st.number_input(
                    f"Flowrate (Nm³/h) for {stream_name}",
                    min_value=0.0,
                    max_value=float(spec['total_flowrate_Nm3h'] * 1.5),
                    value=float(plant_data[stream_name]['total_flowrate_Nm3h']),
                    step=10.0,
                    format="%.1f",
                    key=f"flow_{stream_name}"
                )
            with col2:
                plant_data[stream_name]['T_in_C'] = st.number_input(
                    f"Inlet Temp (°C) for {stream_name}",
                    min_value=-200.0,
                    max_value=100.0,
                    value=float(plant_data[stream_name]['T_in_C']),
                    step=0.1,
                    format="%.1f",
                    key=f"Tin_{stream_name}"
                )
            with col3:
                plant_data[stream_name]['T_out_C'] = st.number_input(
                    f"Outlet Temp (°C) for {stream_name}",
                    min_value=-200.0,
                    max_value=100.0,
                    value=float(plant_data[stream_name]['T_out_C']),
                    step=0.1,
                    format="%.1f",
                    key=f"Tout_{stream_name}"
                )
            # Update vapor/liquid flowrates proportionally for calculation
            flow_ratio = plant_data[stream_name]['total_flowrate_Nm3h'] / spec['total_flowrate_Nm3h'] if spec['total_flowrate_Nm3h'] != 0 else 0
            for key_suffix in ['vapor_in_Nm3h', 'vapor_out_Nm3h', 'liquid_in_Nm3h', 'liquid_out_Nm3h']:
                original_key = key_suffix #f'{key_suffix}_{stream_name}'
                if original_key in spec:
                    plant_data[stream_name][original_key] = spec[original_key] * flow_ratio


# ============================================================
# Design Baseline (from Cell 6)
# ============================================================
# Calculate design effectiveness for baseline
# This part needs to be computed once to set the baseline

if 'design_baseline' not in st.session_state:
    st.session_state.design_baseline = {}

    # Temporarily compute design results using original design_specs
    design_results = {}
    for name, spec in design_specs.items():
        design_results[name] = calculate_heat_duty_enthalpy(name, spec)

    # Overall effectiveness for design
    design_case = compute_efficiency_for_case(design_specs)
    eps_design = design_case['eps_overall']
    Q_hot_design = design_case['Q_hot_total']
    Q_cold_design = design_case['Q_cold_total']

    design_baseline = {
        'eps_overall': eps_design,
        'Q_hot_total_kcalh': Q_hot_design,
        'Q_cold_total_kcalh': Q_cold_design,
        'per_stream': {}
    }

    for name, spec in design_specs.items():
        res = design_results[name]
        design_baseline['per_stream'][name] = {
            'Q_design_kcalh': spec['heat_duty_kcalh'],
            'Q_calc_kcalh': res['Q_kcalh'], # Q_calc is Q_design as per Cell 4 output
            'T_in_C': spec['T_in_C'],
            'T_out_C': spec['T_out_C'],
            'flowrate_Nm3h': spec['total_flowrate_Nm3h'],
            'pressure_drop_mbar': spec['pressure_drop_mbar'],
            'P_operating_bara': spec['P_operating_bara']
        }
    st.session_state.design_baseline = design_baseline

# Retrieve design_baseline from session_state
design_baseline = st.session_state.design_baseline

# ============================================================
# Analysis & Results
# ============================================================

st.header("2. Performance Analysis")

df_comparison, eps_plant, overall_eff = analyze_plant_vs_design(plant_data, design_specs, design_baseline)

# Display Overall Efficiency
st.subheader("Overall Heat Exchanger Efficiency")
col_eff1, col_eff2, col_eff3 = st.columns(3)
with col_eff1:
    st.metric(label="Design Overall Effectiveness (ε)", value=f"{design_baseline['eps_overall']:.4f}")
with col_eff2:
    st.metric(label="Plant Overall Effectiveness (ε)", value=f"{eps_plant:.4f}")
with col_eff3:
    st.metric(label="Overall Efficiency (% of Design)", value=f"{overall_eff:.2f}%", delta=f"{overall_eff - 100:+.2f}%")

# Display Detailed Per-Stream Comparison
st.subheader("Detailed Per-Stream Comparison (Plant vs. Design)")
st.dataframe(df_comparison[[
    'Stream', 'Type', 
    'Flow_plant', 'Flow_design', 'Flow_dev (%)',
    'Tin_plant', 'Tin_design', 'Tin_dev (°C)',
    'Tout_plant', 'Tout_design', 'Tout_dev (°C)',
    'Q_plant (kcal/h)', 'Q_design (kcal/h)', 'Duty_dev (%)'
]])

# ============================================================
# Visualizations
# ============================================================

st.header("3. Visualizations")

import matplotlib.pyplot as plt

streams = list(design_specs.keys())

# Plot 1: Per-stream Heat Duty Comparison
st.subheader("Heat Duty: Plant vs. Design")
fig1, ax1 = plt.subplots(figsize=(10, 5))

q_design_vals = [design_baseline['per_stream'][s]['Q_design_kcalh'] / 1e6 for s in streams]
q_plant_vals = [df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0] / 1e6 if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0]) else 0 for s in streams]

x = np.arange(len(streams))
width = 0.35
ax1.bar(x - width/2, q_design_vals, width, label='Design', color='steelblue', alpha=0.8)
ax1.bar(x + width/2, q_plant_vals, width, label='Plant (Current)', color='coral', alpha=0.8)
ax1.set_xlabel('Stream')
ax1.set_ylabel('Heat Duty (x10^6 kcal/h)')
ax1.set_title('Per-Stream Heat Duty Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(streams)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
st.pyplot(fig1)


# Plot 2: Per-stream Duty Deviation
st.subheader("Per-Stream Heat Duty Deviation from Design")
fig2, ax2 = plt.subplots(figsize=(10, 5))

duty_devs = [df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0] if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]) else 0 for s in streams]

# Color bars based on deviation
bar_colors = []
for dev in duty_devs:
    if abs(dev) < 5: # Within 5%
        bar_colors.append('green')
    elif abs(dev) < 10: # Within 10%
        bar_colors.append('orange')
    else:
        bar_colors.append('red')

ax2.bar(streams, duty_devs, color=bar_colors, alpha=0.8)
ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax2.axhline(5, color='green', linestyle=':', alpha=0.6, label='±5% acceptable')
ax2.axhline(-5, color='green', linestyle=':', alpha=0.6)
ax2.axhline(10, color='orange', linestyle=':', alpha=0.6, label='±10% warning')
ax2.axhline(-10, color='orange', linestyle=':', alpha=0.6)

ax2.set_xlabel('Stream')
ax2.set_ylabel('Duty Deviation (%)')
ax2.set_title('Heat Duty Deviation (%) from Design for Each Stream')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
st.pyplot(fig2)


st.markdown("--- Jardar Singh, 2024 ---")
