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
    delta_T_max = (
        max(plant_data[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'hot') -
        min(plant_data[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'cold')
    )
    delta_T_max_design = (
        max(design_specs[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'hot') -
        min(design_specs[s]['T_in_C'] for s in streams if design_specs[s]['stream_type'] == 'cold')
    )
    eps_plant_per_stream, eps_design_per_stream = {}, {}
    for s in streams:
        dT_p = abs(plant_data[s]['T_out_C']  - plant_data[s]['T_in_C'])
        dT_d = abs(design_specs[s]['T_out_C'] - design_specs[s]['T_in_C'])
        eps_plant_per_stream[s]  = round(dT_p / delta_T_max,        4) if delta_T_max        > 0 else 0
        eps_design_per_stream[s] = round(dT_d / delta_T_max_design, 4) if delta_T_max_design > 0 else 0

    duty_eff, duty_devs = {}, {}
    for s in streams:
        q_p = df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0]
        q_d = df_comparison[df_comparison['Stream'] == s]['Q_design (kcal/h)'].values[0]
        duty_eff[s]  = round((q_p / q_d) * 100, 2) if (q_p and q_d) else 0
        duty_devs[s] = df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0] \
                       if not pd.isna(df_comparison[df_comparison['Stream'] == s]['Duty_dev (%)'].values[0]) else 0

    sorted_streams = sorted(streams, key=lambda s: abs(duty_devs[s]), reverse=True)

    def band_color(val, good=5, warn=10):
        if abs(val) < good:   return '#2ca02c'
        elif abs(val) < warn: return '#ff7f0e'
        else:                 return '#d62728'

    # ════════════════════════════════════════════════════════════
    # ROW 1: Gauge (Overall ε) + Radar (multi-KPI spider)
    # ════════════════════════════════════════════════════════════
    st.subheader("Overall Health & Multi-KPI Comparison")
    col_g, col_r = st.columns([1, 1.6])

    # ── VIZ 1: Gauge — Overall HX Efficiency ────────────────────
    with col_g:
        st.caption("Overall HX Efficiency — Gauge")

        fig_g, ax_g = plt.subplots(figsize=(5, 3.2),
                                   subplot_kw=dict(aspect='equal'))
        ax_g.set_xlim(-1.3, 1.3)
        ax_g.set_ylim(-0.2, 1.3)
        ax_g.axis('off')

        # Draw coloured arc bands (semicircle)
        import matplotlib.patches as mpatches
        from matplotlib.patches import Wedge

        cx, cy, r_out, r_in = 0, 0, 1.1, 0.65

        # Band definitions: (start_deg, end_deg, color, label)
        bands = [
            (180, 216, '#d62728', '<60%'),
            (216, 234, '#ff7f0e', '60–70%'),
            (234, 252, '#ffdd57', '70–80%'),
            (252, 279, '#a8d5a2', '80–90%'),
            (279, 297, '#4caf50', '90–95%'),
            (297, 360, '#1b7c3d', '95–100%'),
        ]
        for (t1, t2, col, _) in bands:
            wedge = Wedge((cx, cy), r_out, t1, t2,
                          width=r_out - r_in, color=col, alpha=0.9)
            ax_g.add_patch(wedge)

        # Tick marks & labels
        tick_pcts = [0, 20, 40, 60, 80, 100]
        for pct in tick_pcts:
            angle_deg = 180 - pct * 1.8
            angle_rad = np.radians(angle_deg)
            x1 = cx + r_in  * np.cos(angle_rad)
            y1 = cy + r_in  * np.sin(angle_rad)
            x2 = cx + (r_in - 0.07) * np.cos(angle_rad)
            y2 = cy + (r_in - 0.07) * np.sin(angle_rad)
            ax_g.plot([x1, x2], [y1, y2], color='white', lw=1.2)
            xt = cx + (r_in - 0.18) * np.cos(angle_rad)
            yt = cy + (r_in - 0.18) * np.sin(angle_rad)
            ax_g.text(xt, yt, f'{pct}%', ha='center', va='center',
                      fontsize=6.5, color='#333333')

        # Needle
        needle_pct  = min(max(overall_eff, 0), 100)
        needle_deg  = 180 - needle_pct * 1.8
        needle_rad  = np.radians(needle_deg)
        needle_len  = r_in - 0.05
        ax_g.annotate('',
            xy=(cx + needle_len * np.cos(needle_rad),
                cy + needle_len * np.sin(needle_rad)),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle='->', color='#1a1a2e',
                            lw=2.5, mutation_scale=14))
        # Centre pivot
        circle = plt.Circle((cx, cy), 0.06, color='#1a1a2e', zorder=5)
        ax_g.add_patch(circle)

        # Central value text
        eff_col = '#1b7c3d' if overall_eff >= 97 else \
                  '#856404' if overall_eff >= 90 else '#d62728'
        ax_g.text(cx, cy - 0.22, f'{overall_eff:.1f}%',
                  ha='center', va='top', fontsize=18,
                  fontweight='bold', color=eff_col)
        ax_g.text(cx, cy - 0.44, 'of Design',
                  ha='center', va='top', fontsize=8, color='#555555')

        status = 'GOOD' if overall_eff >= 97 else \
                 'WARNING' if overall_eff >= 90 else 'CRITICAL'
        ax_g.text(cx, -0.15, status,
                  ha='center', va='top', fontsize=10,
                  fontweight='bold', color=eff_col,
                  bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='#f5f5f5', edgecolor=eff_col, lw=1.5))

        fig_g.tight_layout()
        st.pyplot(fig_g)

    # ── VIZ 2: Radar / Spider — Multi-KPI per Stream ─────────────
    with col_r:
        st.caption("Multi-KPI Spider — Plant vs Design per Stream")

        # Metrics normalised 0→1: duty_eff/100, eps, flow_ratio
        # Each stream is one spoke; 3 rings = 3 KPIs
        kpi_labels = ['Duty Eff.\n(% of Design)', 'Effectiveness\n(ε ratio)',
                      'Flow Ratio\n(plant/design)']
        N_kpi = len(kpi_labels)
        angles = np.linspace(0, 2 * np.pi, len(streams), endpoint=False).tolist()
        angles += angles[:1]   # close polygon

        plant_duty   = [duty_eff[s] / 100 for s in streams]
        design_duty  = [1.0] * len(streams)
        plant_eps    = [eps_plant_per_stream[s]  /
                        max(eps_design_per_stream[s], 0.001) for s in streams]
        design_eps   = [1.0] * len(streams)
        plant_flow   = [plant_data[s]['total_flowrate_Nm3h'] /
                        design_specs[s]['total_flowrate_Nm3h'] for s in streams]
        design_flow  = [1.0] * len(streams)

        # One radar per KPI metric, 3 panels side by side
        fig_r, axes_r = plt.subplots(1, 3, figsize=(9, 3.8),
                                     subplot_kw=dict(polar=True))

        kpi_data = [
            (plant_duty,  design_duty,  '#ED7D31', '#4472C4', 'Duty Eff.'),
            (plant_eps,   design_eps,   '#ED7D31', '#4472C4', 'ε Ratio'),
            (plant_flow,  design_flow,  '#ED7D31', '#4472C4', 'Flow Ratio'),
        ]

        for ax_r, (p_vals, d_vals, pc, dc, title) in zip(axes_r, kpi_data):
            p_plot = p_vals + p_vals[:1]
            d_plot = d_vals + d_vals[:1]

            ax_r.plot(angles, d_plot, color=dc, lw=1.5,
                      linestyle='--', label='Design')
            ax_r.fill(angles, d_plot, color=dc, alpha=0.08)
            ax_r.plot(angles, p_plot, color=pc, lw=2,
                      linestyle='-',  label='Plant')
            ax_r.fill(angles, p_plot, color=pc, alpha=0.18)

            ax_r.set_xticks(angles[:-1])
            ax_r.set_xticklabels(streams, fontsize=7.5)
            ax_r.set_yticklabels([])
            ax_r.set_title(title, fontsize=9, fontweight='bold', pad=12)
            ax_r.legend(fontsize=6.5, loc='upper right',
                        bbox_to_anchor=(1.35, 1.1))

            # Annotate plant value at each spoke
            for ang, val, s in zip(angles[:-1], p_vals, streams):
                ax_r.text(ang, val + 0.05, f'{val:.2f}',
                          ha='center', va='bottom', fontsize=6, color=pc)

        fig_r.suptitle('Plant vs Design — KPI Radar per Stream',
                       fontsize=10, fontweight='bold', y=1.02)
        fig_r.tight_layout()
        st.pyplot(fig_r)

    st.divider()

    # ════════════════════════════════════════════════════════════
    # ROW 2: Heatmap — Deviation Matrix
    # ════════════════════════════════════════════════════════════
    st.subheader("Deviation Heatmap — All Parameters × All Streams")
    st.caption("Red = over-design  |  Blue = under-design  |  White = on-target")

    params = ['Flow Dev (%)', 'Tin Dev (°C)', 'Tout Dev (°C)', 'Duty Dev (%)']
    param_keys = ['Flow_dev (%)', 'Tin_dev (°C)', 'Tout_dev (°C)', 'Duty_dev (%)']

    matrix = []
    for pk in param_keys:
        row = []
        for s in streams:
            val = df_comparison[df_comparison['Stream'] == s][pk].values[0]
            row.append(val if val is not None and not pd.isna(val) else 0)
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    fig_h, ax_h = plt.subplots(figsize=(11, 4))

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdBu_r
    # Symmetric scale around 0
    abs_max = max(abs(matrix).max(), 1)
    norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    im = ax_h.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    # Cell text annotations
    for i in range(len(params)):
        for j in range(len(streams)):
            val = matrix[i, j]
            txt_col = 'white' if abs(val) > abs_max * 0.55 else '#111111'
            ax_h.text(j, i, f'{val:+.2f}',
                      ha='center', va='center',
                      fontsize=9.5, fontweight='bold', color=txt_col)

    ax_h.set_xticks(range(len(streams)))
    ax_h.set_xticklabels(
        [f"{s}\n({design_specs[s]['stream_type'].upper()})"
         for s in streams], fontsize=9)
    ax_h.set_yticks(range(len(params)))
    ax_h.set_yticklabels(params, fontsize=9.5)
    ax_h.set_title('Parameter Deviation Matrix — Plant vs Design',
                   fontsize=11, fontweight='bold', pad=12)

    cbar = fig_h.colorbar(im, ax=ax_h, orientation='vertical',
                          fraction=0.03, pad=0.03)
    cbar.set_label('Deviation (units as labelled)', fontsize=8)

    # Stream type separator line
    hot_count = sum(1 for s in streams if design_specs[s]['stream_type'] == 'hot')
    ax_h.axvline(hot_count - 0.5, color='#333333', lw=2, linestyle='--')
    ax_h.text(hot_count - 0.5, -0.7, '← HOT | COLD →',
              ha='center', va='top', fontsize=8, color='#333333')

    fig_h.tight_layout()
    st.pyplot(fig_h)

    st.divider()

    # ════════════════════════════════════════════════════════════
    # ROW 3: Bubble chart + Ranked health bar
    # ════════════════════════════════════════════════════════════
    col_b, col_h = st.columns(2)

    # ── VIZ 4: Bubble Chart — Flow vs ε, sized by heat duty ──────
    with col_b:
        st.subheader("Bubble Chart")
        st.caption("X = Flowrate  |  Y = Effectiveness (ε)  |  Size = Heat Duty  |  Color = Duty Deviation")

        fig_b, ax_b = plt.subplots(figsize=(7, 5.5))

        for s in streams:
            flow = plant_data[s]['total_flowrate_Nm3h']
            eps  = eps_plant_per_stream[s]
            q_p  = df_comparison[df_comparison['Stream'] == s]['Q_plant (kcal/h)'].values[0]
            dev  = duty_devs[s]
            size = max((q_p / 1e5) ** 1.1, 80) if q_p else 80
            col  = band_color(dev)
            stype_marker = 'o' if design_specs[s]['stream_type'] == 'hot' else 's'

            ax_b.scatter(flow, eps, s=size, c=col, marker=stype_marker,
                         alpha=0.82, edgecolors='white', linewidths=1.5, zorder=3)

            # Design position as faint cross
            d_flow = design_specs[s]['total_flowrate_Nm3h']
            d_eps  = eps_design_per_stream[s]
            ax_b.scatter(d_flow, d_eps, s=size * 0.5, c=col,
                         marker='+', alpha=0.45, linewidths=1.5, zorder=2)
            ax_b.plot([d_flow, flow], [d_eps, eps],
                      color=col, lw=0.8, linestyle=':', alpha=0.5, zorder=1)

            # Label with offset to avoid overlap
            ax_b.annotate(s, xy=(flow, eps), fontsize=8,
                          fontweight='bold', color='#111111',
                          xytext=(8, 6), textcoords='offset points')

        # Legend for markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#888',
                   markersize=9,  label='Hot stream (●)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#888',
                   markersize=9,  label='Cold stream (■)'),
            Line2D([0], [0], marker='+', color='#888',
                   markersize=9,  label='Design position (+)'),
            Line2D([0], [0], color='#2ca02c', lw=2, label='< ±5% OK'),
            Line2D([0], [0], color='#ff7f0e', lw=2, label='±5–10% Warning'),
            Line2D([0], [0], color='#d62728', lw=2, label='> ±10% Critical'),
        ]
        ax_b.legend(handles=legend_elements, fontsize=7,
                    loc='upper center', bbox_to_anchor=(0.5, -0.14),
                    ncol=3, framealpha=0.9)

        ax_b.set_xlabel('Plant Flowrate (Nm³/h)', fontsize=10, labelpad=6)
        ax_b.set_ylabel('Plant Effectiveness (ε)', fontsize=10)
        ax_b.set_title('Flowrate vs Effectiveness\n(bubble size ∝ heat duty)',
                       fontsize=10, fontweight='bold', pad=10)
        ax_b.grid(alpha=0.2, zorder=0)
        fig_b.tight_layout(rect=[0, 0.1, 1, 1])
        st.pyplot(fig_b)

    # ── VIZ 5: Ranked horizontal bar — Stream Health Scorecard ───
    with col_h:
        st.subheader("Stream Health Scorecard")
        st.caption("Streams ranked worst → best by duty deviation")

        fig_s, ax_s = plt.subplots(figsize=(7, 5.5))

        y_pos    = np.arange(len(sorted_streams))
        h_devs   = [duty_devs[s] for s in sorted_streams]
        h_effs   = [duty_eff[s]  for s in sorted_streams]
        h_colors = [band_color(d) for d in h_devs]

        bars_s = ax_s.barh(y_pos, h_devs, color=h_colors,
                           alpha=0.88, edgecolor='white',
                           linewidth=0.8, height=0.5)

        x_max = max(abs(d) for d in h_devs) if h_devs else 10
        for bar, dev, eff in zip(bars_s, h_devs, h_effs):
            pad   = x_max * 0.04
            xpos  = dev + pad  if dev >= 0 else dev - pad
            align = 'left'     if dev >= 0 else 'right'
            ax_s.text(xpos, bar.get_y() + bar.get_height() / 2,
                      f'{dev:+.1f}%  |  {eff:.1f}% duty',
                      va='center', ha=align,
                      fontsize=8, fontweight='bold', color='#111111')

        ax_s.set_yticks(y_pos)
        ax_s.set_yticklabels(
            [f"{s}  [{design_specs[s]['stream_type'].upper()}]"
             for s in sorted_streams], fontsize=9.5)
        ax_s.set_xlim(-x_max * 1.65, x_max * 1.65)

        ax_s.axvline(0,   color='#444', lw=1.2, linestyle='--')
        ax_s.axvline(5,   color='#2ca02c', lw=1.0, linestyle=':', alpha=0.8)
        ax_s.axvline(-5,  color='#2ca02c', lw=1.0, linestyle=':', alpha=0.8)
        ax_s.axvline(10,  color='#ff7f0e', lw=1.0, linestyle=':', alpha=0.8)
        ax_s.axvline(-10, color='#ff7f0e', lw=1.0, linestyle=':', alpha=0.8)
        ax_s.axvspan(-5,   5,  alpha=0.07, color='green',  zorder=0)
        ax_s.axvspan( 5,  10,  alpha=0.07, color='orange', zorder=0)
        ax_s.axvspan(-10, -5,  alpha=0.07, color='orange', zorder=0)

        from matplotlib.lines import Line2D
        leg_s = [
            Line2D([0],[0], color='#2ca02c', lw=1.5, linestyle=':', label='±5%  OK'),
            Line2D([0],[0], color='#ff7f0e', lw=1.5, linestyle=':', label='±10% Warning'),
        ]
        ax_s.legend(handles=leg_s, fontsize=8,
                    loc='upper center', bbox_to_anchor=(0.5, -0.1),
                    ncol=2, framealpha=0.9)

        ax_s.set_xlabel('Heat Duty Deviation (%)', fontsize=10, labelpad=8)
        ax_s.set_title('Stream Health Scorecard\nRanked by Severity',
                       fontsize=10, fontweight='bold', pad=10)
        ax_s.grid(axis='x', alpha=0.2, zorder=0)
        fig_s.tight_layout(rect=[0, 0.08, 1, 1])
        st.pyplot(fig_s)
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
