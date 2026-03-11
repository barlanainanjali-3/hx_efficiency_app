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

st.title("🔥 Plate-Fin Heat Exchanger Efficiency Model")
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
# Streamlit UI — Inputs
# ============================================================

st.header("1. Enter Current Plant Operating Data")
st.markdown("Adjust the sliders or type in the values for each stream's flowrate and temperatures.")

plant_data = copy.deepcopy(design_specs)

hot_streams = [s for s, spec in design_specs.items() if spec['stream_type'] == 'hot']
cold_streams = [s for s, spec in design_specs.items() if spec['stream_type'] == 'cold']
hot_tab, cold_tab = st.tabs(["Hot Streams", "Cold Streams"])

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


# ============================================================
# Analysis & Results
# ============================================================

st.header("2. Performance Analysis")
df_comparison, eps_plant, overall_eff = analyze_plant_vs_design(plant_data, design_specs, design_baseline)

st.subheader("Overall Heat Exchanger Efficiency")
col_eff1, col_eff2, col_eff3 = st.columns(3)
with col_eff1:
    st.metric(label="Design Overall Effectiveness (ε)", value=f"{design_baseline['eps_overall']:.4f}")
with col_eff2:
    st.metric(label="Plant Overall Effectiveness (ε)", value=f"{eps_plant:.4f}")
with col_eff3:
    st.metric(label="Overall Efficiency (% of Design)", value=f"{overall_eff:.2f}%", delta=f"{overall_eff - 100:+.2f}%")

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

streams = list(design_specs.keys())

# Plot 1
st.subheader("Heat Duty: Plant vs. Design")
fig1, ax1 = plt.subplots(figsize=(10, 4.5))
q_design_vals = [design_baseline['per_stream'][s]['Q_design_kcalh'] / 1e6 for s in streams]
q_plant_vals = [
    df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0] / 1e6
    if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0]) else 0
    for s in streams
]
x = np.arange(len(streams))
width = 0.35
ax1.bar(x - width/2, q_design_vals, width, label='Design', color='steelblue', alpha=0.85)
ax1.bar(x + width/2, q_plant_vals, width, label='Plant (Current)', color='coral', alpha=0.85)
ax1.set_xlabel('Stream')
ax1.set_ylabel('Heat Duty (×10⁶ kcal/h)')
ax1.set_title('Per-Stream Heat Duty Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(streams)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
fig1.tight_layout()
st.pyplot(fig1)

# Plot 2
st.subheader("Per-Stream Heat Duty Deviation from Design")
fig2, ax2 = plt.subplots(figsize=(10, 4.5))
duty_devs = [
    df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]
    if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]) else 0
    for s in streams
]
bar_colors = []
for dev in duty_devs:
    if abs(dev) < 5:
        bar_colors.append('#2ca02c')
    elif abs(dev) < 10:
        bar_colors.append('#ff7f0e')
    else:
        bar_colors.append('#d62728')

ax2.bar(streams, duty_devs, color=bar_colors, alpha=0.85)
ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax2.axhline(5, color='green', linestyle=':', alpha=0.6, label='±5% acceptable')
ax2.axhline(-5, color='green', linestyle=':', alpha=0.6)
ax2.axhline(10, color='orange', linestyle=':', alpha=0.6, label='±10% warning')
ax2.axhline(-10, color='orange', linestyle=':', alpha=0.6)
ax2.set_xlabel('Stream')
ax2.set_ylabel('Duty Deviation (%)')
ax2.set_title('Heat Duty Deviation (%) from Design')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
fig2.tight_layout()
st.pyplot(fig2)


# ============================================================
# Auto Root Cause + Manual Notes
# ============================================================

st.header("4. Root Cause & Suggested Adjustments")

auto_root_causes, auto_suggestions = generate_root_cause_and_suggestions(df_comparison, overall_eff)

with st.expander("📋 Auto-Generated Root Cause Analysis", expanded=True):
    for line in auto_root_causes:
        if '[CRITICAL]' in line:
            st.error(line)
        elif '[WARNING]' in line:
            st.warning(line)
        elif '[OK]' in line:
            st.success(line)
        else:
            st.write(line)

with st.expander("🔧 Auto-Generated Suggested Adjustments", expanded=True):
    for i, sug in enumerate(auto_suggestions, 1):
        st.info(f"**{i}.** {sug}")

st.subheader("✏️ Additional Engineer Notes")
manual_notes = st.text_area(
    "Add your own observations, manual root cause, or additional recommendations here:",
    height=150,
    placeholder="e.g. Observed frost buildup on AGMP stream outlet. Valve HV-101 was found partially closed during inspection...",
    key="manual_notes"
)


# ============================================================
# PDF Export
# ============================================================

st.header("5. Download Report")
st.markdown("Download a full PDF report including all graphs, metrics, root cause analysis, and your notes.")

if st.button("📄 Generate & Download PDF Report", type="primary"):
    report_timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")

    # Capture fig1 as bytes
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
    buf1.seek(0)
    fig1_bytes = buf1.read()

    # Capture fig2 as bytes
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    buf2.seek(0)
    fig2_bytes = buf2.read()

    with st.spinner("Building PDF report..."):
        pdf_bytes = generate_pdf_report(
            df_comparison=df_comparison,
            overall_eff=overall_eff,
            eps_plant=eps_plant,
            eps_design=design_baseline['eps_overall'],
            root_causes=auto_root_causes,
            suggestions=auto_suggestions,
            manual_notes=manual_notes,
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
    st.success(f"Report ready: **{file_name}**")


st.markdown("--- Jardar Singh, 2024 ---")
