import streamlit as st
import numpy as np
import pandas as pd
import copy
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from CoolProp.CoolProp import PropsSI

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak
)
from reportlab.platypus.flowables import KeepTogether

# ============================================================
# Streamlit App: Heat Exchanger Efficiency Model
# ============================================================

st.set_page_config(
    page_title="HX Efficiency Model",
    page_icon="🔥",
    layout="wide"
)

st.title("Plate-Fin Heat Exchanger Efficiency Model")
st.markdown("### Multi-Stream Counter-Flow — JSPL India (ASU)")
st.write("Adjust the parameters below to analyze real-time plant performance against design specifications.")

# ============================================================
# Static Design Specifications
# ============================================================

FLUID_MAP = {
    'AGMP': 'Air',
    'AGHP': 'Air',
    'AGT':  'Air',
    'GOX':  'Oxygen',
    'GAN':  'Nitrogen',
    'WN2':  'Nitrogen'
}

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
        'phase_change': True,
        'stream_type': 'hot'
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
        'phase_change': True,
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
        'phase_change': True,
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
        'phase_change': True,
        'stream_type': 'cold'
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
# Core Functions
# ============================================================

def C_to_K(T_C):
    return T_C + 273.15

def bara_to_Pa(P_bara):
    return P_bara * 1e5

def Nm3h_to_kgs(flowrate_Nm3h, fluid):
    T_normal = 273.15
    P_normal = 101325
    rho_normal = PropsSI('D', 'T', T_normal, 'P', P_normal, fluid)
    return (flowrate_Nm3h * rho_normal) / 3600.0

def get_specific_enthalpy(fluid, T_C, P_bara, phase='gas'):
    T_K = C_to_K(T_C)
    P_Pa = bara_to_Pa(P_bara)
    try:
        return PropsSI('H', 'T', T_K, 'P', P_Pa, fluid)
    except Exception:
        try:
            if phase == 'liquid':
                return PropsSI('H', 'P', P_Pa, 'Q', 0, fluid)
            else:
                return PropsSI('H', 'P', P_Pa, 'Q', 1, fluid)
        except Exception:
            return None

def calculate_heat_duty_enthalpy(stream_name, spec):
    fluid = spec['fluid']
    T_in = spec['T_in_C']
    T_out = spec['T_out_C']
    P_bara = spec['P_operating_bara']
    total_flow = spec['total_flowrate_Nm3h']
    m_dot = Nm3h_to_kgs(total_flow, fluid)
    details = []

    vap_in = spec['vapor_in_Nm3h']
    liq_in = spec['liquid_in_Nm3h']

    if liq_in > 0 and vap_in == 0:
        h_in = get_specific_enthalpy(fluid, T_in, P_bara, phase='liquid')
        details.append(f"Inlet: pure liquid at {T_in}°C")
    elif liq_in > 0 and vap_in > 0:
        frac_vap = vap_in / total_flow
        frac_liq = liq_in / total_flow
        h_vap = get_specific_enthalpy(fluid, T_in, P_bara, phase='gas')
        h_liq = get_specific_enthalpy(fluid, T_in, P_bara, phase='liquid')
        h_in = frac_vap * h_vap + frac_liq * h_liq
        details.append(f"Inlet: mixed (vap={frac_vap:.2f}, liq={frac_liq:.2f})")
    else:
        h_in = get_specific_enthalpy(fluid, T_in, P_bara, phase='gas')
        details.append(f"Inlet: pure vapor at {T_in}°C")

    vap_out = spec['vapor_out_Nm3h']
    liq_out = spec['liquid_out_Nm3h']

    if liq_out > 0 and vap_out == 0:
        h_out = get_specific_enthalpy(fluid, T_out, P_bara, phase='liquid')
        details.append(f"Outlet: pure liquid at {T_out}°C")
    elif liq_out > 0 and vap_out > 0:
        frac_vap = vap_out / total_flow
        frac_liq = liq_out / total_flow
        h_vap = get_specific_enthalpy(fluid, T_out, P_bara, phase='gas')
        h_liq = get_specific_enthalpy(fluid, T_out, P_bara, phase='liquid')
        h_out = frac_vap * h_vap + frac_liq * h_liq
        details.append(f"Outlet: mixed (vap={frac_vap:.2f}, liq={frac_liq:.2f})")
    else:
        h_out = get_specific_enthalpy(fluid, T_out, P_bara, phase='gas')
        details.append(f"Outlet: pure vapor at {T_out}°C")

    if h_in is not None and h_out is not None:
        Q_watts = m_dot * abs(h_out - h_in)
        Q_kcalh = Q_watts * 3600 / 4184
    else:
        Q_watts = None
        Q_kcalh = None
        details.append("Could not compute enthalpy")

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

def compute_efficiency_for_case(case_specs):
    results = {}
    for name, spec in case_specs.items():
        results[name] = calculate_heat_duty_enthalpy(name, spec)

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

def analyze_plant_vs_design(plant_data, design_specs, design_baseline):
    plant_case = compute_efficiency_for_case(plant_data)
    comparison = []
    for name in design_specs.keys():
        d = design_baseline['per_stream'][name]
        p_result = plant_case['results'][name]
        p_spec = plant_data[name]

        flow_dev = ((p_spec['total_flowrate_Nm3h'] - d['flowrate_Nm3h']) / d['flowrate_Nm3h'] * 100)
        T_in_dev = p_spec['T_in_C'] - d['T_in_C']
        T_out_dev = p_spec['T_out_C'] - d['T_out_C']
        q_plant = p_result['Q_kcalh']
        q_design = d['Q_design_kcalh']
        duty_dev = ((q_plant - q_design) / q_design * 100) if q_plant else None
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
    eps_plant = plant_case['eps_overall']
    eps_design = design_baseline['eps_overall']
    overall_efficiency = (eps_plant / eps_design) * 100
    return df_comp, eps_plant, overall_efficiency


# ============================================================
# Auto Root Cause & Suggested Adjustments
# ============================================================

def generate_root_cause_and_suggestions(df_comparison, overall_eff):
    """Auto-generate root cause analysis and suggested adjustments from deviations."""
    root_causes = []
    suggestions = []

    for _, row in df_comparison.iterrows():
        stream = row['Stream']
        stype = row['Type'].upper()
        duty_dev = row['Duty_dev (%)'] if row['Duty_dev (%)'] is not None else 0
        flow_dev = row['Flow_dev (%)']
        tin_dev = row['Tin_dev (°C)']
        tout_dev = row['Tout_dev (°C)']

        stream_issues = []
        stream_suggestions = []

        # Duty deviation checks
        if abs(duty_dev) > 10:
            severity = "CRITICAL"
            stream_issues.append(
                f"[{severity}] {stream} ({stype}): Heat duty deviation of {duty_dev:+.1f}% "
                f"— significantly outside ±10% tolerance band."
            )
        elif abs(duty_dev) > 5:
            severity = "WARNING"
            stream_issues.append(
                f"[{severity}] {stream} ({stype}): Heat duty deviation of {duty_dev:+.1f}% "
                f"— within warning band (5–10%)."
            )

        # Flow deviation checks
        if abs(flow_dev) > 5:
            direction = "lower" if flow_dev < 0 else "higher"
            stream_issues.append(
                f"  → Flow {direction} than design by {abs(flow_dev):.1f}% "
                f"({row['Flow_plant']:.0f} vs {row['Flow_design']:.0f} Nm³/h)."
            )
            if flow_dev < -5:
                stream_suggestions.append(
                    f"{stream}: Investigate upstream supply pressure or valve restrictions "
                    f"causing reduced flowrate. Check control valve position and line strainers."
                )
            else:
                stream_suggestions.append(
                    f"{stream}: Excess flow may overload downstream separation. "
                    f"Review feed compressor throughput and adjust setpoint."
                )

        # Outlet temperature deviation
        if abs(tout_dev) > 3:
            direction = "warmer" if tout_dev > 0 else "colder"
            stream_issues.append(
                f"  → Outlet temperature {direction} than design by {abs(tout_dev):.1f}°C "
                f"({row['Tout_plant']:.1f}°C vs {row['Tout_design']:.1f}°C)."
            )
            if stype == 'HOT' and tout_dev > 3:
                stream_suggestions.append(
                    f"{stream}: Insufficient cooling — check cold stream flow balance and "
                    f"heat exchanger fouling. Inspect for maldistribution or partial blockage."
                )
            elif stype == 'HOT' and tout_dev < -3:
                stream_suggestions.append(
                    f"{stream}: Over-cooling detected — may indicate higher cold stream flow "
                    f"or lower hot side inlet temperature. Review stream balance."
                )
            elif stype == 'COLD' and tout_dev < -3:
                stream_suggestions.append(
                    f"{stream}: Under-heating on cold side — verify hot stream conditions and "
                    f"check for bypass or leakage paths in the HX core."
                )

        # Inlet temperature deviation
        if abs(tin_dev) > 2:
            stream_issues.append(
                f"  → Inlet temperature deviation of {tin_dev:+.1f}°C from design "
                f"({row['Tin_plant']:.1f}°C vs {row['Tin_design']:.1f}°C)."
            )

        if stream_issues:
            root_causes.extend(stream_issues)
        if stream_suggestions:
            suggestions.extend(stream_suggestions)

    # Overall efficiency root cause
    if overall_eff < 90:
        root_causes.insert(0,
            f"[CRITICAL] Overall HX efficiency is {overall_eff:.1f}% — below 90% threshold. "
            f"Immediate investigation required."
        )
        suggestions.insert(0,
            "Schedule urgent HX performance audit. Inspect for fouling, maldistribution, "
            "and process imbalances across all streams."
        )
    elif overall_eff < 97:
        root_causes.insert(0,
            f"[WARNING] Overall HX efficiency is {overall_eff:.1f}% — moderately below design. "
            f"Monitor closely."
        )
        suggestions.insert(0,
            "Review combined stream heat balance. Consider scheduled maintenance inspection "
            "within next planned shutdown window."
        )
    else:
        root_causes.insert(0,
            f"[OK] Overall HX efficiency is {overall_eff:.1f}% — within acceptable design range."
        )
        suggestions.insert(0,
            "No immediate action required. Continue routine monitoring per SOPs."
        )

    if not root_causes[1:]:
        root_causes.append("All individual stream parameters are within acceptable tolerance bands.")
    if not suggestions[1:]:
        suggestions.append("Maintain current operating conditions. No stream-level adjustments needed.")

    return root_causes, suggestions


# ============================================================
# PDF Generation Function
# ============================================================

def generate_pdf_report(
    df_comparison,
    overall_eff,
    eps_plant,
    eps_design,
    root_causes,
    suggestions,
    manual_notes,
    fig1_bytes,
    fig2_bytes,
    report_timestamp
):
    buffer = io.BytesIO()
    PAGE_W, PAGE_H = A4
    MARGIN = 2 * cm

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="HX Efficiency Report — JSPL India ASU"
    )

    # ---- Custom Styles ----
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    style_subtitle = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#4a4a6a'),
        spaceAfter=2,
        alignment=TA_CENTER
    )
    style_timestamp = ParagraphStyle(
        'Timestamp',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#888888'),
        spaceAfter=16,
        alignment=TA_CENTER
    )
    style_section = ParagraphStyle(
        'Section',
        parent=styles['Heading1'],
        fontSize=13,
        textColor=colors.HexColor('#1a1a2e'),
        spaceBefore=16,
        spaceAfter=8,
        borderPad=4,
        fontName='Helvetica-Bold'
    )
    style_body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#222222')
    )
    style_bullet = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        leftIndent=12,
        textColor=colors.HexColor('#333333')
    )
    style_notes = ParagraphStyle(
        'Notes',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#444444'),
        backColor=colors.HexColor('#f9f9f9'),
        borderPad=8
    )

    story = []
    content_width = PAGE_W - 2 * MARGIN

    # ============================================================
    # HEADER
    # ============================================================
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("🔥 Heat Exchanger Efficiency Report", style_title))
    story.append(Paragraph("Plate-Fin Multi-Stream Counter-Flow — JSPL India (ASU)", style_subtitle))
    story.append(Paragraph(f"Generated: {report_timestamp}", style_timestamp))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#1a1a2e'), spaceAfter=12))

    # ============================================================
    # SECTION 1: LARGE METRIC CARDS — Efficiency & Deviation
    # ============================================================
    story.append(Paragraph("1. Overall Performance Summary", style_section))

    def eff_color(eff):
        if eff >= 97:
            return colors.HexColor('#1b7c3d'), colors.HexColor('#d4edda'), "● GOOD"
        elif eff >= 90:
            return colors.HexColor('#856404'), colors.HexColor('#fff3cd'), "▲ WARNING"
        else:
            return colors.HexColor('#721c24'), colors.HexColor('#f8d7da'), "✖ CRITICAL"

    txt_col, bg_col, status_label = eff_color(overall_eff)

    # Big metric card: Overall Efficiency
    card_data = [
        [
            Paragraph(
                f'<font size="11" color="#555555">Design Effectiveness (ε)</font>',
                style_body
            ),
            Paragraph(
                f'<font size="11" color="#555555">Plant Effectiveness (ε)</font>',
                style_body
            ),
            Paragraph(
                f'<font size="11" color="#555555">Overall Efficiency vs Design</font>',
                style_body
            ),
        ],
        [
            Paragraph(
                f'<font size="26" color="#1a1a2e"><b>{eps_design:.4f}</b></font>',
                ParagraphStyle('card', alignment=TA_CENTER)
            ),
            Paragraph(
                f'<font size="26" color="#1a1a2e"><b>{eps_plant:.4f}</b></font>',
                ParagraphStyle('card', alignment=TA_CENTER)
            ),
            Paragraph(
                f'<font size="32"><b><font color="{txt_col.hexval()}">{overall_eff:.1f}%</font></b></font>'
                f'<br/><font size="11" color="{txt_col.hexval()}">{status_label}</font>',
                ParagraphStyle('card', alignment=TA_CENTER)
            ),
        ],
    ]

    col_w = content_width / 3
    card_table = Table(card_data, colWidths=[col_w, col_w, col_w])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, -1), colors.HexColor('#f0f4ff')),
        ('BACKGROUND', (2, 0), (2, -1), bg_col),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#cccccc')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS', [6], ),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 0.5 * cm))

    # ============================================================
    # Per-Stream Deviation Cards
    # ============================================================
    story.append(Paragraph("Per-Stream Duty Deviation", style_section))

    def deviation_color(dev):
        if dev is None:
            return colors.HexColor('#dddddd'), colors.black, "N/A"
        if abs(dev) < 5:
            return colors.HexColor('#d4edda'), colors.HexColor('#155724'), f"{dev:+.1f}%"
        elif abs(dev) < 10:
            return colors.HexColor('#fff3cd'), colors.HexColor('#856404'), f"{dev:+.1f}%"
        else:
            return colors.HexColor('#f8d7da'), colors.HexColor('#721c24'), f"{dev:+.1f}%"

    streams_list = df_comparison['Stream'].tolist()
    # 3 cards per row
    row_cards = []
    all_card_rows = []
    for i, stream in enumerate(streams_list):
        row_data = df_comparison[df_comparison['Stream'] == stream].iloc[0]
        dev = row_data['Duty_dev (%)']
        bg, fg, dev_label = deviation_color(dev)
        q_plant = row_data['Q_plant (kcal/h)']
        q_design = row_data['Q_design (kcal/h)']
        stype = row_data['Type'].upper()

        cell = Table(
            [
                [Paragraph(f'<b><font size="13">{stream}</font></b><br/>'
                           f'<font size="8" color="#666666">{stype}</font>',
                           ParagraphStyle('cc', alignment=TA_CENTER))],
                [Paragraph(f'<font size="22"><b><font color="{fg.hexval()}">{dev_label}</font></b></font>',
                           ParagraphStyle('cc', alignment=TA_CENTER))],
                [Paragraph(f'<font size="7.5" color="#555555">'
                           f'Plant: {q_plant:,.0f} kcal/h<br/>'
                           f'Design: {q_design:,.0f} kcal/h</font>',
                           ParagraphStyle('cc', alignment=TA_CENTER))],
            ],
            colWidths=[(content_width / 3) - 4 * mm]
        )
        cell.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bbbbbb')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        row_cards.append(cell)
        if len(row_cards) == 3:
            all_card_rows.append(row_cards)
            row_cards = []
    if row_cards:
        while len(row_cards) < 3:
            row_cards.append(Paragraph("", style_body))
        all_card_rows.append(row_cards)

    card_col_w = content_width / 3
    for card_row in all_card_rows:
        t = Table([card_row], colWidths=[card_col_w, card_col_w, card_col_w])
        t.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 3),
            ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 2 * mm))

    story.append(Spacer(1, 0.3 * cm))

    # Legend
    legend_data = [[
        Paragraph('<font size="8" color="#155724">■ Green: Within ±5% (OK)</font>', style_body),
        Paragraph('<font size="8" color="#856404">■ Orange: ±5–10% (Warning)</font>', style_body),
        Paragraph('<font size="8" color="#721c24">■ Red: >±10% (Critical)</font>', style_body),
    ]]
    legend_t = Table(legend_data, colWidths=[content_width / 3] * 3)
    legend_t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f5f5')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(legend_t)
    story.append(Spacer(1, 0.4 * cm))

    # ============================================================
    # SECTION 2: DETAILED COMPARISON TABLE
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Detailed Per-Stream Comparison", style_section))

    table_cols = ['Stream', 'Type', 'Flow\nPlant', 'Flow\nDesign', 'Flow\nDev(%)',
                  'Tin\nPlant(°C)', 'Tin\nDes(°C)', 'Tout\nPlant(°C)', 'Tout\nDes(°C)',
                  'Q Plant\n(kcal/h)', 'Q Design\n(kcal/h)', 'Duty\nDev(%)']

    header_row = [Paragraph(f'<b><font size="7">{c}</font></b>', ParagraphStyle('th', alignment=TA_CENTER))
                  for c in table_cols]

    table_rows = [header_row]
    for _, row in df_comparison.iterrows():
        dev = row['Duty_dev (%)']
        bg_c, fg_c, _ = deviation_color(dev)

        def fmt(val, dec=1):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:.{dec}f}"

        data_row = [
            Paragraph(f'<b><font size="7">{row["Stream"]}</font></b>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{row["Type"]}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Flow_plant"],0)}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Flow_design"],0)}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Flow_dev (%)"])}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Tin_plant"])}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Tin_design"])}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Tout_plant"])}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Tout_design"])}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Q_plant (kcal/h)"],0)}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7">{fmt(row["Q_design (kcal/h)"],0)}</font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
            Paragraph(f'<font size="7"><b>{fmt(dev)}</b></font>',
                      ParagraphStyle('td', alignment=TA_CENTER)),
        ]
        table_rows.append(data_row)

    col_widths = [
        1.4*cm, 1.0*cm, 1.5*cm, 1.5*cm, 1.4*cm,
        1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm,
        2.0*cm, 2.0*cm, 1.5*cm
    ]
    comp_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    row_bg_styles = []
    for i, (_, row) in enumerate(df_comparison.iterrows(), start=1):
        dev = row['Duty_dev (%)']
        bg_c, _, _ = deviation_color(dev)
        row_bg_styles.append(('BACKGROUND', (0, i), (-1, i), bg_c))

    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#999999')),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        *row_bg_styles
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 0.5 * cm))

    # ============================================================
    # SECTION 3: GRAPHS
    # ============================================================
    story.append(Paragraph("3. Performance Graphs", style_section))

    fig_width = content_width
    fig_height = 7 * cm

    img1 = RLImage(io.BytesIO(fig1_bytes), width=fig_width, height=fig_height)
    story.append(KeepTogether([
        Paragraph("Heat Duty: Plant vs. Design (per stream)", style_body),
        Spacer(1, 2 * mm),
        img1
    ]))
    story.append(Spacer(1, 0.4 * cm))

    img2 = RLImage(io.BytesIO(fig2_bytes), width=fig_width, height=fig_height)
    story.append(KeepTogether([
        Paragraph("Heat Duty Deviation (%) from Design", style_body),
        Spacer(1, 2 * mm),
        img2
    ]))
    story.append(Spacer(1, 0.4 * cm))

    # ============================================================
    # SECTION 4: ROOT CAUSE ANALYSIS
    # ============================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Root Cause Analysis", style_section))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc'), spaceAfter=8))

    for line in root_causes:
        # Color-code by severity prefix
        if '[CRITICAL]' in line:
            color_hex = '#721c24'
            bg_hex = '#f8d7da'
        elif '[WARNING]' in line:
            color_hex = '#856404'
            bg_hex = '#fff3cd'
        elif '[OK]' in line:
            color_hex = '#155724'
            bg_hex = '#d4edda'
        else:
            color_hex = '#333333'
            bg_hex = '#f5f5f5'

        p = Paragraph(
            f'<font color="{color_hex}">{line}</font>',
            ParagraphStyle('rc', parent=style_bullet, backColor=colors.HexColor(bg_hex),
                           borderPad=6, spaceBefore=3, spaceAfter=3,
                           borderWidth=0, leading=14)
        )
        story.append(p)

    story.append(Spacer(1, 0.4 * cm))

    # ============================================================
    # SECTION 5: SUGGESTED ADJUSTMENTS
    # ============================================================
    story.append(Paragraph("5. Suggested Adjustments", style_section))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc'), spaceAfter=8))

    for i, sug in enumerate(suggestions, 1):
        p = Paragraph(
            f'<b>{i}.</b> {sug}',
            ParagraphStyle('sug', parent=style_bullet,
                           backColor=colors.HexColor('#eef4ff'),
                           borderPad=6, spaceBefore=3, spaceAfter=3, leading=14)
        )
        story.append(p)

    story.append(Spacer(1, 0.5 * cm))

    # ============================================================
    # SECTION 6: MANUAL NOTES
    # ============================================================
    story.append(Paragraph("6. Engineer Notes", style_section))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc'), spaceAfter=8))

    notes_text = manual_notes.strip() if manual_notes.strip() else "(No additional notes provided.)"
    # Preserve line breaks
    notes_text_html = notes_text.replace('\n', '<br/>')
    notes_para = Paragraph(
        notes_text_html,
        ParagraphStyle('notes', parent=style_notes,
                       backColor=colors.HexColor('#f9f9f9'),
                       borderColor=colors.HexColor('#cccccc'),
                       borderWidth=1, borderPad=10)
    )
    # Box it
    notes_table = Table([[notes_para]], colWidths=[content_width])
    notes_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#aaaaaa')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f9f9f9')),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(notes_table)

    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
    story.append(Paragraph(
        f'<font size="8" color="#888888">JSPL India ASU — HX Efficiency Report | {report_timestamp} | Generated by HX Efficiency Model</font>',
        ParagraphStyle('footer', alignment=TA_CENTER, spaceBefore=6)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# Design Baseline (computed once)
# ============================================================

if 'design_baseline' not in st.session_state:
    st.session_state.design_baseline = {}
    design_results = {}
    for name, spec in design_specs.items():
        design_results[name] = calculate_heat_duty_enthalpy(name, spec)
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
            'Q_calc_kcalh': res['Q_kcalh'],
            'T_in_C': spec['T_in_C'],
            'T_out_C': spec['T_out_C'],
            'flowrate_Nm3h': spec['total_flowrate_Nm3h'],
            'pressure_drop_mbar': spec['pressure_drop_mbar'],
            'P_operating_bara': spec['P_operating_bara']
        }
    st.session_state.design_baseline = design_baseline

design_baseline = st.session_state.design_baseline

# ============================================================
# Streamlit UI — Main Tabs
# ============================================================
plant_data = copy.deepcopy(design_specs)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥  1. Plant Inputs",
    "📊  2. Performance Analysis",
    "📈  3. Visualizations",
    "🔍  4. Root Cause",
    "📄  5. Download Report"
])

# ============================================================
# TAB 1 — Plant Inputs
# ============================================================
with tab1:
    st.header("Enter Current Plant Operating Data")
    st.markdown("Adjust the values for each stream's flowrate and temperatures.")

    hot_streams = [s for s, spec in design_specs.items() if spec['stream_type'] == 'hot']
    cold_streams = [s for s, spec in design_specs.items() if spec['stream_type'] == 'cold']

    hot_tab, cold_tab = st.tabs(["🔴 Hot Streams", "🔵 Cold Streams"])

    with hot_tab:
        st.subheader("Hot Streams (Cooling)")
        for stream_name in hot_streams:
            spec = design_specs[stream_name]
            with st.expander(f"**{stream_name}** (Design: {spec['T_in_C']}°C in, {spec['T_out_C']}°C out, {spec['total_flowrate_Nm3h']} Nm³/h)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    plant_data[stream_name]['total_flowrate_Nm3h'] = st.number_input(
                        f"Flowrate (Nm³/h) for {stream_name}",
                        min_value=0.0, max_value=float(spec['total_flowrate_Nm3h'] * 1.5),
                        value=float(plant_data[stream_name]['total_flowrate_Nm3h']),
                        step=10.0, format="%.1f", key=f"flow_{stream_name}")
                with col2:
                    plant_data[stream_name]['T_in_C'] = st.number_input(
                        f"Inlet Temp (°C) for {stream_name}",
                        min_value=-200.0, max_value=100.0,
                        value=float(plant_data[stream_name]['T_in_C']),
                        step=0.1, format="%.1f", key=f"Tin_{stream_name}")
                with col3:
                    plant_data[stream_name]['T_out_C'] = st.number_input(
                        f"Outlet Temp (°C) for {stream_name}",
                        min_value=-200.0, max_value=100.0,
                        value=float(plant_data[stream_name]['T_out_C']),
                        step=0.1, format="%.1f", key=f"Tout_{stream_name}")
                flow_ratio = plant_data[stream_name]['total_flowrate_Nm3h'] / spec['total_flowrate_Nm3h'] if spec['total_flowrate_Nm3h'] != 0 else 0
                for key_suffix in ['vapor_in_Nm3h', 'vapor_out_Nm3h', 'liquid_in_Nm3h', 'liquid_out_Nm3h']:
                    if key_suffix in spec:
                        plant_data[stream_name][key_suffix] = spec[key_suffix] * flow_ratio

    with cold_tab:
        st.subheader("Cold Streams (Heating)")
        for stream_name in cold_streams:
            spec = design_specs[stream_name]
            with st.expander(f"**{stream_name}** (Design: {spec['T_in_C']}°C in, {spec['T_out_C']}°C out, {spec['total_flowrate_Nm3h']} Nm³/h)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    plant_data[stream_name]['total_flowrate_Nm3h'] = st.number_input(
                        f"Flowrate (Nm³/h) for {stream_name}",
                        min_value=0.0, max_value=float(spec['total_flowrate_Nm3h'] * 1.5),
                        value=float(plant_data[stream_name]['total_flowrate_Nm3h']),
                        step=10.0, format="%.1f", key=f"flow_{stream_name}")
                with col2:
                    plant_data[stream_name]['T_in_C'] = st.number_input(
                        f"Inlet Temp (°C) for {stream_name}",
                        min_value=-200.0, max_value=100.0,
                        value=float(plant_data[stream_name]['T_in_C']),
                        step=0.1, format="%.1f", key=f"Tin_{stream_name}")
                with col3:
                    plant_data[stream_name]['T_out_C'] = st.number_input(
                        f"Outlet Temp (°C) for {stream_name}",
                        min_value=-200.0, max_value=100.0,
                        value=float(plant_data[stream_name]['T_out_C']),
                        step=0.1, format="%.1f", key=f"Tout_{stream_name}")
                flow_ratio = plant_data[stream_name]['total_flowrate_Nm3h'] / spec['total_flowrate_Nm3h'] if spec['total_flowrate_Nm3h'] != 0 else 0
                for key_suffix in ['vapor_in_Nm3h', 'vapor_out_Nm3h', 'liquid_in_Nm3h', 'liquid_out_Nm3h']:
                    if key_suffix in spec:
                        plant_data[stream_name][key_suffix] = spec[key_suffix] * flow_ratio

    st.info("✅ After entering values, navigate to the other tabs to view results.")


# ============================================================
# Run analysis (shared across all tabs)
# ============================================================
df_comparison, eps_plant, overall_eff = analyze_plant_vs_design(plant_data, design_specs, design_baseline)
auto_root_causes, auto_suggestions = generate_root_cause_and_suggestions(df_comparison, overall_eff)


# ============================================================
# TAB 2 — Performance Analysis
# ============================================================
with tab2:
    st.header("Performance Analysis")

    # Overall efficiency color
    if overall_eff >= 97:
        eff_color_fn = st.success
    elif overall_eff >= 90:
        eff_color_fn = st.warning
    else:
        eff_color_fn = st.error

    eff_color_fn(f"Overall Efficiency vs Design: **{overall_eff:.2f}%**  (Δ {overall_eff - 100:+.2f}%)")

    st.subheader("Overall Effectiveness")
    col_eff1, col_eff2, col_eff3 = st.columns(3)
    with col_eff1:
        st.metric(label="Design Overall Effectiveness (ε)", value=f"{design_baseline['eps_overall']:.4f}")
    with col_eff2:
        st.metric(label="Plant Overall Effectiveness (ε)", value=f"{eps_plant:.4f}")
    with col_eff3:
        st.metric(label="Overall Efficiency (% of Design)", value=f"{overall_eff:.2f}%",
                  delta=f"{overall_eff - 100:+.2f}%")

    st.divider()
    st.subheader("Detailed Per-Stream Comparison (Plant vs. Design)")
    st.dataframe(df_comparison[[
        'Stream', 'Type',
        'Flow_plant', 'Flow_design', 'Flow_dev (%)',
        'Tin_plant', 'Tin_design', 'Tin_dev (°C)',
        'Tout_plant', 'Tout_design', 'Tout_dev (°C)',
        'Q_plant (kcal/h)', 'Q_design (kcal/h)', 'Duty_dev (%)'
    ]], use_container_width=True)


# ============================================================
# TAB 3 — Visualizations
# ============================================================
with tab3:
    st.header("Visualizations")
    streams = list(design_specs.keys())

    # ── Shared data prep ────────────────────────────────────────
    # Per-stream effectiveness (plant)
    delta_T_max = (
        max(plant_data[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'hot') -
        min(plant_data[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'cold')
    )
    eps_plant_per_stream = {}
    for s in streams:
        dT = abs(plant_data[s]['T_out_C'] - plant_data[s]['T_in_C'])
        eps_plant_per_stream[s] = round(dT / delta_T_max, 4) if delta_T_max > 0 else 0

    # Per-stream effectiveness (design)
    delta_T_max_design = (
        max(design_specs[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'hot') -
        min(design_specs[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'cold')
    )
    eps_design_per_stream = {}
    for s in streams:
        dT = abs(design_specs[s]['T_out_C'] - design_specs[s]['T_in_C'])
        eps_design_per_stream[s] = round(dT / delta_T_max_design, 4) if delta_T_max_design > 0 else 0

    # Heat duty efficiency % per stream (Q_plant / Q_design * 100)
    duty_eff = {}
    for s in streams:
        q_p = df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0]
        q_d = df_comparison[df_comparison['Stream'] == s]['Q_design (kcal/h)'].values[0]
        duty_eff[s] = round((q_p / q_d) * 100, 2) if (q_p and q_d) else 0

    # Duty deviation %
    duty_devs = {
        s: df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]
        if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]) else 0
        for s in streams
    }

    # Sort streams by absolute duty deviation for Graph 1
    sorted_streams = sorted(streams, key=lambda s: abs(duty_devs[s]), reverse=True)

    def band_color(val, good=5, warn=10):
        """Return fill color based on deviation magnitude."""
        if abs(val) < good:
            return '#2ca02c'
        elif abs(val) < warn:
            return '#ff7f0e'
        else:
            return '#d62728'

    # ── GRAPH 1: Stream Health Scorecard (full width) ────────────
    st.subheader("Stream Health Scorecard — Underperformance Ranking")
    st.caption("Streams sorted by severity. Red = critical (>±10%), Orange = warning (5–10%), Green = OK (<±5%)")

    # ── GRAPH 1: Stream Health Scorecard (full width) ────────────
    st.subheader("Stream Health Scorecard — Underperformance Ranking")
    st.caption("Sorted by severity. Red = critical (>±10%), Orange = warning (5–10%), Green = OK (<±5%)")

    fig1, ax1 = plt.subplots(figsize=(13, 6))

    y_pos    = np.arange(len(sorted_streams))
    h_devs   = [duty_devs[s]  for s in sorted_streams]
    h_effs   = [duty_eff[s]   for s in sorted_streams]
    h_colors = [band_color(d) for d in h_devs]

    bars = ax1.barh(y_pos, h_devs, color=h_colors, alpha=0.88,
                    edgecolor='white', linewidth=0.8, height=0.5)

    # Fix 1: push label outside bar with a fixed padding, never overlap bar
    x_max = max(abs(d) for d in h_devs) if h_devs else 10
    for bar, dev, eff in zip(bars, h_devs, h_effs):
        pad   = x_max * 0.03
        xpos  = dev + pad  if dev >= 0 else dev - pad
        align = 'left'     if dev >= 0 else 'right'
        ax1.text(xpos, bar.get_y() + bar.get_height() / 2,
                 f'{dev:+.1f}%   Duty: {eff:.1f}%',
                 va='center', ha=align, fontsize=8.5,
                 fontweight='bold', color='#111111')

    ytick_labels = [
        f"{s}  [{design_specs[s]['stream_type'].upper()}]"
        for s in sorted_streams
    ]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(ytick_labels, fontsize=10)

    # Extend x-axis so labels don't get clipped
    ax1.set_xlim(-x_max * 1.55, x_max * 1.55)

    ax1.axvline(0,   color='#444444', linewidth=1.2, linestyle='--')
    ax1.axvline(5,   color='#2ca02c', linewidth=1.0, linestyle=':', alpha=0.8)
    ax1.axvline(-5,  color='#2ca02c', linewidth=1.0, linestyle=':', alpha=0.8)
    ax1.axvline(10,  color='#ff7f0e', linewidth=1.0, linestyle=':', alpha=0.8)
    ax1.axvline(-10, color='#ff7f0e', linewidth=1.0, linestyle=':', alpha=0.8)
    ax1.axvspan(-5,   5,  alpha=0.07, color='green',  zorder=0)
    ax1.axvspan( 5,  10,  alpha=0.07, color='orange', zorder=0)
    ax1.axvspan(-10, -5,  alpha=0.07, color='orange', zorder=0)

    oc = '#1b7c3d' if overall_eff >= 97 else '#856404' if overall_eff >= 90 else '#721c24'
    ax1.annotate(
        f'Overall HX Efficiency\n{overall_eff:.1f}% of Design',
        xy=(0.99, 0.04), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=10, fontweight='bold', color=oc,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5',
                  edgecolor=oc, linewidth=2)
    )

    ax1.set_xlabel('Heat Duty Deviation from Design (%)', fontsize=11, labelpad=8)
    ax1.set_title(
        'Stream Health Scorecard — Heat Duty Deviation (%) Ranked by Severity',
        fontsize=11, fontweight='bold', pad=12
    )
# move legend outside right of Graph 1
    from matplotlib.lines import Line2D
    legend_elements1 = [
        Line2D([0], [0], color='#2ca02c', lw=1.5, linestyle=':', label='±5%  Acceptable'),
        Line2D([0], [0], color='#ff7f0e', lw=1.5, linestyle=':', label='±10% Warning'),
    ]
    ax1.legend(handles=legend_elements1, fontsize=8,
               loc='upper left', framealpha=0.9,
               bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax1.set_xlabel('Heat Duty Deviation from Design (%)', fontsize=11, labelpad=8)
    ax1.set_title(
        'Stream Health Scorecard — Heat Duty Deviation (%) Ranked by Severity',
        fontsize=11, fontweight='bold', pad=12
    )
    ax1.grid(axis='x', alpha=0.2, zorder=0)
    fig1.tight_layout(rect=[0, 0, 0.88, 1])
    st.pyplot(fig1)

    st.divider()

    col_left, col_right = st.columns(2)

    # ── GRAPH 2: Heat Duty Efficiency % ─────────────────────────
    with col_left:
        st.subheader("Heat Duty Efficiency % per Stream")
        st.caption("How much of the design duty each stream is delivering")

        fig2, ax2 = plt.subplots(figsize=(7, 6))

        eff_vals   = [duty_eff[s] for s in streams]
        bar_colors = [band_color(v - 100, good=5, warn=10) for v in eff_vals]

        bars2 = ax2.bar(streams, eff_vals, color=bar_colors, alpha=0.85,
                        edgecolor='white', linewidth=0.8, zorder=2, width=0.5)

        ax2.axhline(100, color='#1a1a2e', linewidth=2.0, linestyle='--',
                    label='Design (100%)', zorder=3)
        ax2.axhline(95,  color='#ff7f0e', linewidth=1.0, linestyle=':',
                    alpha=0.7, label='95% warning', zorder=3)
        ax2.axhline(90,  color='#d62728', linewidth=1.0, linestyle=':',
                    alpha=0.7, label='90% critical', zorder=3)
        ax2.axhspan(95, 110, alpha=0.07, color='green',  zorder=0)
        ax2.axhspan(90,  95, alpha=0.07, color='orange', zorder=0)
        ax2.axhspan(75,  90, alpha=0.07, color='red',    zorder=0)

        # Fix 2: efficiency % on top, Q value below that — no inside-bar text
        y_min = max(0, min(eff_vals) - 18)
        for bar, val, s in zip(bars2, eff_vals, streams):
            # Efficiency % — just above bar top
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.6,
                     f'{val:.1f}%',
                     ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color='#111111')
            # Q plant value — below bar near baseline, never overlapping %
            q_p = df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0]
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     y_min + 1.0,
                     f'{q_p/1e6:.2f}M',
                     ha='center', va='bottom',
                     fontsize=7, color='#333333')

        # Fix 3: stream type as rotated x-tick labels — no extra text() calls
        ax2.set_xticks(range(len(streams)))
        ax2.set_xticklabels(
            [f"{s}\n({design_specs[s]['stream_type'].upper()})"
             for s in streams],
            fontsize=8.5
        )

        ax2.set_ylim(bottom=y_min, top=max(eff_vals) + 14)
        ax2.set_xlabel('Stream', fontsize=10, labelpad=6)
        ax2.set_ylabel('Duty Efficiency (% of Design)', fontsize=10)
        ax2.set_title('Heat Duty Efficiency %\nvs Design Baseline',
                      fontsize=10, fontweight='bold', pad=10)
       ax2.set_ylim(bottom=y_min, top=max(eff_vals) + 14)
    ax2.set_xlabel('Stream', fontsize=10, labelpad=6)
    ax2.set_ylabel('Duty Efficiency (% of Design)', fontsize=10)
    ax2.set_title('Heat Duty Efficiency %\nvs Design Baseline',
                  fontsize=10, fontweight='bold', pad=10)
    # legend below the chart — never overlaps bars
    ax2.legend(fontsize=7.5, loc='upper center',
               bbox_to_anchor=(0.5, -0.18),
               ncol=3, framealpha=0.9, borderaxespad=0)
    ax2.grid(axis='y', alpha=0.2, zorder=0)
    fig2.tight_layout(rect=[0, 0.08, 1, 1])
    st.pyplot(fig2)
    # ── GRAPH 3: Per-Stream Effectiveness (ε) ───────────────────
    with col_right:
        st.subheader("Per-Stream Effectiveness (ε)")
        st.caption("Temperature-based effectiveness: plant vs design per stream")

        fig3, ax3 = plt.subplots(figsize=(7, 6))

        x3    = np.arange(len(streams))
        w3    = 0.35
        eps_d = [eps_design_per_stream[s] for s in streams]
        eps_p = [eps_plant_per_stream[s]  for s in streams]

        bars_d3 = ax3.bar(x3 - w3/2, eps_d, w3, label='Design ε',
                          color='#4472C4', alpha=0.85, zorder=2)
        bars_p3 = ax3.bar(x3 + w3/2, eps_p, w3, label='Plant ε',
                          color='#ED7D31', alpha=0.85, zorder=2)

        # Fix 4: stagger design/plant ε labels vertically to avoid collision
        for bar, val in zip(bars_d3, eps_d):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.004,
                     f'{val:.3f}',
                     ha='center', va='bottom',
                     fontsize=7.5, color='#1a3a5c', fontweight='bold')

        for bar, val in zip(bars_p3, eps_p):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.004,
                     f'{val:.3f}',
                     ha='center', va='bottom',
                     fontsize=7.5, color='#7b3300', fontweight='bold')

        # Δε label placed above the taller of the two bars with extra clearance
        top_eps = max(max(eps_d), max(eps_p))
        for i, s in enumerate(streams):
            delta_e = eps_p[i] - eps_d[i]
            color   = '#155724' if abs(delta_e) < 0.02 else \
                      '#856404' if abs(delta_e) < 0.05 else '#721c24'
            bar_top = max(eps_d[i], eps_p[i])
            ax3.annotate(
                f'Δε\n{delta_e:+.3f}',
                xy=(i, bar_top + 0.008),
                xytext=(i, bar_top + 0.045),
                ha='center', va='bottom', fontsize=7, color=color,
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=0.7)
            )

        ax3.axhline(design_baseline['eps_overall'], color='#4472C4',
                    linewidth=1.5, linestyle='--', alpha=0.6,
                    label=f"Design avg ε={design_baseline['eps_overall']:.3f}")
        ax3.axhline(eps_plant, color='#ED7D31',
                    linewidth=1.5, linestyle='--', alpha=0.6,
                    label=f"Plant avg ε ={eps_plant:.3f}")

        ax3.set_xticks(x3)
        ax3.set_xticklabels(
            [f"{s}\n({design_specs[s]['stream_type'].upper()})"
             for s in streams],
            fontsize=8.5
        )
        ax3.set_xlabel('Stream', fontsize=10, labelpad=6)
        ax3.set_ylabel('Effectiveness (ε)', fontsize=10)
        ax3.set_title('Per-Stream Effectiveness (ε)\nPlant vs Design',
                      fontsize=10, fontweight='bold', pad=10)
        ax3.set_xlabel('Stream', fontsize=10, labelpad=6)
    ax3.set_ylabel('Effectiveness (ε)', fontsize=10)
    ax3.set_title('Per-Stream Effectiveness (ε)\nPlant vs Design',
                  fontsize=10, fontweight='bold', pad=10)
    # legend below the chart — never overlaps Δε annotations above bars
    ax3.legend(fontsize=7.5, loc='upper center',
               bbox_to_anchor=(0.5, -0.18),
               ncol=2, framealpha=0.9, borderaxespad=0)
    ax3.set_ylim(bottom=0, top=top_eps + 0.18)
    ax3.grid(axis='y', alpha=0.2, zorder=0)
    fig3.tight_layout(rect=[0, 0.08, 1, 1])
    st.pyplot(fig3)
# ============================================================
# TAB 4 — Root Cause & Suggested Adjustments
# ============================================================
with tab4:
    st.header("Root Cause & Suggested Adjustments")

    st.subheader("📋 Auto-Generated Root Cause Analysis")
    for line in auto_root_causes:
        if '[CRITICAL]' in line:
            st.error(line)
        elif '[WARNING]' in line:
            st.warning(line)
        elif '[OK]' in line:
            st.success(line)
        else:
            st.write(f"&nbsp;&nbsp;&nbsp;{line}")

    st.divider()

    st.subheader("🔧 Suggested Adjustments")
    for i, sug in enumerate(auto_suggestions, 1):
        st.info(f"**{i}.** {sug}")

    st.divider()

    st.subheader("✏️ Additional Engineer Notes")
    manual_notes = st.text_area(
        "Add your own observations, manual root cause, or additional recommendations:",
        height=180,
        placeholder="e.g. Observed frost buildup on AGMP outlet. Valve HV-101 was found partially closed...",
        key="manual_notes"
    )


# ============================================================
# TAB 5 — Download Report
# ============================================================
with tab5:
    st.header("Download PDF Report")
    st.markdown(
        "Generates a full report including metric cards, deviation graphs, "
        "root cause analysis, suggested adjustments, and your engineer notes."
    )

    # Retrieve manual notes safely (may not have been filled if user skips tab 4)
    manual_notes_val = st.session_state.get("manual_notes", "")

    if st.button("📄 Generate & Download PDF Report", type="primary"):
        report_timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")

        # Re-render figures for PDF capture
        _streams = list(design_specs.keys())
        _fig1, _ax1 = plt.subplots(figsize=(10, 4.5))
        _q_design = [design_baseline['per_stream'][s]['Q_design_kcalh'] / 1e6 for s in _streams]
        _q_plant  = [
            df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0] / 1e6
            if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0]) else 0
            for s in _streams
        ]
        _x = np.arange(len(_streams))
        _w = 0.35
        _ax1.bar(_x - _w/2, _q_design, _w, label='Design',         color='steelblue', alpha=0.85)
        _ax1.bar(_x + _w/2, _q_plant,  _w, label='Plant (Current)', color='coral',     alpha=0.85)
        _ax1.set_xticks(_x); _ax1.set_xticklabels(_streams)
        _ax1.set_xlabel('Stream'); _ax1.set_ylabel('Heat Duty (×10⁶ kcal/h)')
        _ax1.set_title('Per-Stream Heat Duty Comparison')
        _ax1.legend(); _ax1.grid(axis='y', alpha=0.3); _fig1.tight_layout()

        _fig2, _ax2 = plt.subplots(figsize=(10, 4.5))
        _duty_devs  = [
            df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]
            if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]) else 0
            for s in _streams
        ]
        _bcolors = ['#2ca02c' if abs(d) < 5 else '#ff7f0e' if abs(d) < 10 else '#d62728' for d in _duty_devs]
        _ax2.bar(_streams, _duty_devs, color=_bcolors, alpha=0.85)
        _ax2.axhline(0,   color='grey',   linestyle='--', linewidth=0.8)
        _ax2.axhline(5,   color='green',  linestyle=':', alpha=0.6, label='±5% acceptable')
        _ax2.axhline(-5,  color='green',  linestyle=':', alpha=0.6)
        _ax2.axhline(10,  color='orange', linestyle=':', alpha=0.6, label='±10% warning')
        _ax2.axhline(-10, color='orange', linestyle=':', alpha=0.6)
        _ax2.set_xlabel('Stream'); _ax2.set_ylabel('Duty Deviation (%)')
        _ax2.set_title('Heat Duty Deviation (%) from Design')
        _ax2.legend(); _ax2.grid(axis='y', alpha=0.3); _fig2.tight_layout()

        buf1 = io.BytesIO()
        _fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
        buf1.seek(0); fig1_bytes = buf1.read()

        buf2 = io.BytesIO()
        _fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
        buf2.seek(0); fig2_bytes = buf2.read()

        plt.close(_fig1); plt.close(_fig2)

        with st.spinner("Building PDF report..."):
            pdf_bytes = generate_pdf_report(
                df_comparison=df_comparison,
                overall_eff=overall_eff,
                eps_plant=eps_plant,
                eps_design=design_baseline['eps_overall'],
                root_causes=auto_root_causes,
                suggestions=auto_suggestions,
                manual_notes=manual_notes_val,
                fig1_bytes=fig1_bytes,
                fig2_bytes=fig2_bytes,
                report_timestamp=report_timestamp
            )

        file_name = f"HX_Efficiency_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(
            label="⬇️ Click here to download your PDF",
            data=pdf_bytes,
            file_name=file_name,
            mime="application/pdf"
        )
        st.success(f"✅ Report ready: **{file_name}**")
