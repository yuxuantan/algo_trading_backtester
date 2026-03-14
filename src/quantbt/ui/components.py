from __future__ import annotations

import html
from typing import Any

import streamlit as st


def _status_label(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _render_status_badge(label: str, ok: bool) -> None:
    status = _status_label(ok)
    if ok:
        border = "#16a34a"
        bg = "rgba(22,163,74,0.10)"
        fg = "#166534"
    else:
        border = "#dc2626"
        bg = "rgba(220,38,38,0.10)"
        fg = "#991b1b"

    st.markdown(
        (
            f'<div style="display:inline-flex;align-items:center;gap:10px;'
            f'padding:7px 12px;border-radius:12px;border:2px solid {border};'
            f'background:{bg};margin:4px 0 10px 0;">'
            f'<span style="font-size:12px;font-weight:700;color:{fg};">{label}</span>'
            f'<span style="font-size:24px;line-height:1;font-weight:900;color:{fg};">{status}</span>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_decision_badge(label: str, decision: str) -> None:
    decision_norm = str(decision).strip().upper()
    if decision_norm == "PASS":
        border = "#16a34a"
        bg = "rgba(22,163,74,0.10)"
        fg = "#166534"
    elif decision_norm == "RETRY":
        border = "#d97706"
        bg = "rgba(217,119,6,0.12)"
        fg = "#92400e"
    else:
        decision_norm = "FAIL"
        border = "#dc2626"
        bg = "rgba(220,38,38,0.10)"
        fg = "#991b1b"

    st.markdown(
        (
            f'<div style="display:inline-flex;align-items:center;gap:10px;'
            f'padding:7px 12px;border-radius:12px;border:2px solid {border};'
            f'background:{bg};margin:4px 0 10px 0;">'
            f'<span style="font-size:12px;font-weight:700;color:{fg};">{label}</span>'
            f'<span style="font-size:24px;line-height:1;font-weight:900;color:{fg};">{decision_norm}</span>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def _inject_app_css() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top right, rgba(22,98,74,0.10), transparent 24%),
              linear-gradient(180deg, #f7f4ed 0%, #f1eee6 100%);
          }
          .block-container {
            max-width: 1480px;
            padding-top: 1rem;
            padding-bottom: 2.2rem;
          }
          h1, h2, h3, h4 {
            letter-spacing: -0.02em;
          }
          h1 {
            font-size: 3rem !important;
            line-height: 1.02 !important;
            margin: 0 0 0.4rem 0 !important;
          }
          h2 {
            font-size: 2.15rem !important;
            line-height: 1.08 !important;
            margin: 1rem 0 0.35rem 0 !important;
          }
          h3 {
            font-size: 1.45rem !important;
            line-height: 1.12 !important;
            margin: 0.85rem 0 0.3rem 0 !important;
          }
          h4 {
            font-size: 1.12rem !important;
            line-height: 1.18 !important;
            margin: 0.7rem 0 0.2rem 0 !important;
          }
          [data-testid="stMetric"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
          }
          div[data-baseweb="select"] > div,
          .stTextInput > div > div > input,
          .stTextArea textarea,
          .stNumberInput input {
            border-radius: 12px !important;
            background: rgba(255,255,255,0.88) !important;
          }
          .qt-metric-card {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(141,129,105,0.20);
            border-radius: 16px;
            padding: 0.8rem 0.9rem 0.85rem 0.9rem;
            min-height: 96px;
            box-shadow: 0 8px 18px rgba(77,55,19,0.04);
          }
          .qt-metric-label {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6b6658;
            margin-bottom: 0.45rem;
          }
          .qt-metric-value {
            font-size: clamp(1.08rem, 1.7vw, 1.85rem);
            line-height: 1.1;
            font-weight: 700;
            color: #313445;
            word-break: break-word;
          }
          .qt-metric-value-long {
            font-size: clamp(0.98rem, 1.35vw, 1.22rem);
            line-height: 1.2;
          }
          .qt-subtle-card {
            border: 1px solid #d9d1c2;
            border-radius: 18px;
            padding: 14px 16px;
            background: rgba(255,255,255,0.82);
            box-shadow: 0 12px 30px rgba(77,55,19,0.05);
            margin: 0 0 10px 0;
          }
          .qt-kicker {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6b6658;
            font-weight: 700;
            margin-bottom: 0.45rem;
          }
          .qt-card-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #1f2a23;
            margin-bottom: 0.3rem;
          }
          .qt-card-note {
            font-size: 0.92rem;
            color: #5c584f;
            margin-bottom: 0;
          }
          .qt-chip-title {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #5b574b;
            margin: 0 0 0.45rem 0;
          }
          .qt-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0 0 0.6rem 0;
          }
          .qt-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            font-size: 0.84rem;
            line-height: 1.25;
            border: 1px solid transparent;
          }
          .qt-chip-input {
            background: rgba(34,102,78,0.08);
            color: #1a5c46;
            border-color: rgba(34,102,78,0.12);
          }
          .qt-chip-output {
            background: rgba(180,83,9,0.09);
            color: #92400e;
            border-color: rgba(180,83,9,0.12);
          }
          .qt-chip-pass {
            background: rgba(22,163,74,0.10);
            color: #166534;
            border-color: rgba(22,163,74,0.14);
          }
          div.stButton > button {
            border-radius: 999px;
            padding: 0.52rem 1rem;
            font-weight: 650;
            border: 1px solid rgba(31,86,63,0.18);
          }
          div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #17624a, #103e35);
            color: white;
            border: none;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_chip_group(title: str, items: list[str], *, tone: str) -> None:
    tone_class = {
        "input": "qt-chip-input",
        "output": "qt-chip-output",
        "pass": "qt-chip-pass",
    }.get(tone, "qt-chip-input")
    chips = [
        f'<span class="qt-chip {tone_class}">{html.escape(str(item).strip())}</span>'
        for item in items
        if str(item).strip()
    ]
    if not chips:
        st.caption("None")
        return
    st.markdown(
        (
            f'<div class="qt-chip-title">{html.escape(title)}</div>'
            f'<div class="qt-chip-row">{"".join(chips)}</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_metric_row(metrics: list[tuple[str, Any, str]]) -> None:
    cols = st.columns(len(metrics)) if metrics else []
    for c, (label, value, fmt) in zip(cols, metrics):
        display = value
        if isinstance(value, (int, float)) and fmt:
            try:
                display = fmt.format(value)
            except Exception:
                display = value
        display_text = str(display)
        value_class = "qt-metric-value qt-metric-value-long" if len(display_text) > 20 else "qt-metric-value"
        c.markdown(
            (
                '<div class="qt-metric-card">'
                f'<div class="qt-metric-label">{html.escape(str(label))}</div>'
                f'<div class="{value_class}">{html.escape(display_text)}</div>'
                "</div>"
            ),
            unsafe_allow_html=True,
        )
