import io
import re
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# PDF export dependencies
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)

st.set_page_config(page_title="Machine Age Intelligence", page_icon="🏭", layout="wide")

BRAND = {
    "navy": "#0B1F3A", "blue": "#1565C0", "cyan": "#00A6A6",
    "green": "#2E7D32", "orange": "#F59E0B", "red": "#C62828",
    "ink": "#172033", "muted": "#667085", "bg": "#F4F7FB", "card": "#FFFFFF"
}
AGE_BANDS = [
    "0–5 years", "6–10 years", "11–15 years", "16–20 years",
    "21–25 years", "26+ years", "Unclassified"
]
AGE_COLORS = {
    "0–5 years": "#2E7D32", "6–10 years": "#66BB6A",
    "11–15 years": "#F9A825", "16–20 years": "#EF6C00",
    "21–25 years": "#D84315", "26+ years": "#8E2430",
    "Unclassified": "#98A2B3"
}

st.markdown(f"""
<style>
.stApp {{background: linear-gradient(180deg,#eef4fb 0,#f8fafc 230px); color:{BRAND['ink']};}}
.block-container {{padding-top:1.15rem; padding-bottom:3rem; max-width:1500px;}}
.hero {{padding:1.3rem 1.5rem; border-radius:20px; background:linear-gradient(120deg,{BRAND['navy']},{BRAND['blue']}); color:white; box-shadow:0 14px 38px rgba(11,31,58,.18); margin-bottom:1rem;}}
.hero h1 {{margin:0;font-size:2rem}} .hero p {{margin:.35rem 0 0;color:#dbeafe}}
.kpi {{background:white;border:1px solid #e6ebf2;border-radius:16px;padding:1rem 1.1rem;box-shadow:0 5px 18px rgba(16,24,40,.06);min-height:112px}}
.kpi .label {{color:{BRAND['muted']};font-size:.78rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}}
.kpi .value {{font-size:1.75rem;font-weight:800;color:{BRAND['navy']};margin-top:.3rem}}
.kpi .hint {{font-size:.78rem;color:{BRAND['muted']};margin-top:.2rem}}
.section-title {{font-size:1.15rem;font-weight:800;color:{BRAND['navy']};margin:.35rem 0 .75rem}}
[data-testid="stSidebar"] {{background:#f8fafc;border-right:1px solid #e5e7eb}}
[data-testid="stDataFrame"] {{border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;box-shadow:0 4px 14px rgba(16,24,40,.04)}}
[data-testid="stFileUploader"] {{background:white;border:1px dashed #94a3b8;border-radius:14px;padding:.35rem}}
[data-testid="stDownloadButton"] button {{min-height:48px;border-radius:12px;font-weight:800;border:0;background:linear-gradient(120deg,#0B1F3A,#1565C0);color:white}}
[data-testid="stDownloadButton"] button:hover {{color:white;border:0;filter:brightness(1.08)}}
div[data-baseweb="tab-list"] {{gap:.4rem;background:#eaf0f7;padding:.35rem;border-radius:12px}}
button[data-baseweb="tab"] {{background:white;border-radius:9px;padding:.65rem 1rem;font-weight:700}}
button[data-baseweb="tab"][aria-selected="true"] {{background:#1565C0;color:white}}
.status-strip {{display:flex;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 1rem}}
.status-pill {{background:#eef6ff;color:#0B1F3A;border:1px solid #cfe1f7;border-radius:999px;padding:.38rem .75rem;font-size:.82rem;font-weight:700}}
@media (max-width: 800px) {{.hero h1{{font-size:1.55rem}}.block-container{{padding-left:.7rem;padding-right:.7rem}}.kpi{{min-height:96px}}}}
</style>
<div class="hero"><h1>🏭 Machine Age Intelligence</h1><p>Unit-wise and category-wise asset age analysis for Kenpark & Regency machine master data</p></div>
""", unsafe_allow_html=True)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def find_col(columns, candidates):
    nmap = {norm(c): c for c in columns}
    for candidate in candidates:
        if norm(candidate) in nmap:
            return nmap[norm(candidate)]
    for candidate in candidates:
        nc = norm(candidate)
        for c in columns:
            if nc and (nc in norm(c) or norm(c) in nc):
                return c
    return None


def parse_date_value(v) -> pd.Timestamp:
    if pd.isna(v) or str(v).strip() == "":
        return pd.NaT
    if isinstance(v, (pd.Timestamp, date, datetime)):
        return pd.Timestamp(v)
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        if 1900 <= float(v) <= 2100:
            return pd.Timestamp(year=int(v), month=1, day=1)
        if 1 <= float(v) <= 80000:
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(v), unit="D")
    text = str(v).strip()
    if re.fullmatch(r"(19|20)\d{2}", text):
        return pd.Timestamp(year=int(text), month=1, day=1)
    for dayfirst in (False, True):
        try:
            parsed = pd.to_datetime(text, errors="raise", dayfirst=dayfirst)
            if 1900 <= parsed.year <= 2100:
                return parsed.normalize()
        except Exception:
            pass
    return pd.NaT


def age_from_text(v) -> Optional[float]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    years = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:year|yr)", s)
    months = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:month|mon)", s)
    if years or months:
        y = float(years.group(1)) if years else 0
        m = float(months.group(1)) if months else 0
        return y + m / 12
    cleaned = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(cleaned)
    except Exception:
        return None


def age_band(age):
    if pd.isna(age) or age < 0:
        return "Unclassified"
    if age < 6:
        return "0–5 years"
    if age < 11:
        return "6–10 years"
    if age < 16:
        return "11–15 years"
    if age < 21:
        return "16–20 years"
    if age < 26:
        return "21–25 years"
    return "26+ years"


def clean_label(v, fallback="Unclassified"):
    if pd.isna(v):
        return fallback
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s.upper() if s else fallback


@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, sheet_name):
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="openpyxl", dtype=object)
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw.dropna(how="all")


def prepare_data(raw, unit_col, category_col, purchase_col, supplied_age_col, qty_col, report_date):
    df = raw.copy()
    df["Unit"] = df[unit_col].map(clean_label)
    df["Category"] = df[category_col].map(clean_label)
    df["Quantity"] = (
        pd.to_numeric(df[qty_col], errors="coerce").fillna(1.0)
        if qty_col else pd.Series(1.0, index=df.index)
    )
    df["Quantity"] = df["Quantity"].clip(lower=0)
    df["Purchase Date Parsed"] = df[purchase_col].map(parse_date_value) if purchase_col else pd.Series(pd.NaT, index=df.index)
    cutoff = pd.Timestamp(report_date)
    date_age = (cutoff - df["Purchase Date Parsed"]).dt.days / 365.2425
    supplied_age = (
        df[supplied_age_col].map(age_from_text)
        if supplied_age_col else pd.Series(np.nan, index=df.index)
    )
    # Purchase date is preferred; supplied age is used only if date is unavailable.
    df["Age Years"] = date_age.where(date_age.notna(), supplied_age)
    df.loc[df["Age Years"] < 0, "Age Years"] = np.nan
    df["Age Whole Years"] = np.floor(df["Age Years"]).astype("Int64")
    df["Age Band"] = df["Age Years"].map(age_band)
    df["Age Band"] = pd.Categorical(df["Age Band"], AGE_BANDS, ordered=True)
    df["Data Status"] = np.select(
        [
            df["Purchase Date Parsed"].notna() & df["Age Years"].notna(),
            df["Purchase Date Parsed"].isna() & df["Age Years"].notna(),
        ],
        ["Calculated from purchase date", "Used supplied age"],
        default="Age unavailable"
    )
    return df


def weighted_average(df):
    good = df["Age Years"].notna() & (df["Quantity"] > 0)
    if not good.any():
        return np.nan
    return float(np.average(df.loc[good, "Age Years"], weights=df.loc[good, "Quantity"]))


def band_summary(df, group_cols):
    if df.empty:
        columns = group_cols + AGE_BANDS + ["Total Machines", "Average Age"]
        return pd.DataFrame(columns=columns)
    out = df.groupby(group_cols + ["Age Band"], observed=False)["Quantity"].sum().unstack(fill_value=0)
    out = out.reindex(columns=AGE_BANDS, fill_value=0).reset_index()
    out["Total Machines"] = out[AGE_BANDS].sum(axis=1)
    average_rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in df.groupby(grouper, observed=True, dropna=False):
        if len(group_cols) == 1:
            keys = (keys,)
        elif not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["Average Age"] = weighted_average(group)
        average_rows.append(row)
    averages = pd.DataFrame(average_rows, columns=group_cols + ["Average Age"])
    return out.merge(averages, on=group_cols, how="left")


def to_excel_bytes(detail, unit_summary, unit_category, overall_exact, overall_bands, quality):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="dd-mmm-yyyy") as writer:
        detail.to_excel(writer, sheet_name="Filtered Detail", index=False)
        unit_summary.to_excel(writer, sheet_name="Unit Age Summary", index=False)
        unit_category.to_excel(writer, sheet_name="Unit Category Age", index=False)
        overall_exact.to_excel(writer, sheet_name="Overall Exact Age", index=False)
        overall_bands.to_excel(writer, sheet_name="Overall Age Bands", index=False)
        quality.to_excel(writer, sheet_name="Data Quality", index=False)
        wb = writer.book
        title = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#0B1F3A", "border": 0})
        date_fmt = wb.add_format({"num_format": "dd-mmm-yyyy"})
        for name, ws in writer.sheets.items():
            ws.freeze_panes(1, 0)
            if ws.dim_colmax is not None:
                ws.autofilter(0, 0, max(1, ws.dim_rowmax or 1), ws.dim_colmax)
                ws.set_column(0, ws.dim_colmax, 16)
            ws.set_row(0, 22, title)
        detail_ws = writer.sheets["Filtered Detail"]
        if detail_ws.dim_colmax is not None:
            detail_ws.set_column(0, min(20, detail_ws.dim_colmax), 18)
        if "Purchase Date Parsed" in detail.columns:
            purchase_date_idx = int(detail.columns.get_loc("Purchase Date Parsed"))
            detail_ws.set_column(purchase_date_idx, purchase_date_idx, 14, date_fmt)
        for sheet_name in ["Unit Age Summary", "Unit Category Age", "Overall Exact Age", "Overall Age Bands"]:
            ws = writer.sheets[sheet_name]
            if ws.dim_colmax is not None:
                ws.set_column(0, ws.dim_colmax, 17)
    return output.getvalue()


def pdf_cell(value):
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, date, datetime)):
        return pd.Timestamp(value).strftime("%d-%b-%Y")
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}" if not float(value).is_integer() else f"{value:,.0f}"
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pdf_table(dataframe, max_rows=60, font_size=7):
    if dataframe is None or dataframe.empty:
        dataframe = pd.DataFrame({"Information": ["No records available"]})
    view = dataframe.head(max_rows).copy()
    data = [[pdf_cell(c) for c in view.columns]]
    data += [[pdf_cell(v) for v in row] for row in view.itertuples(index=False, name=None)]
    page_width = landscape(A4)[0] - 24 * mm
    ncols = max(1, len(data[0]))
    table = Table(data, repeatRows=1, colWidths=[page_width / ncols] * ncols)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F3A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def create_pdf_report(filtered, unit_summary, unit_category, exact, overall_bands, quality,
                      issues, report_date, selected_units, selected_cats, age_filter, search):
    """Create a downloadable, filter-aware PDF report with charts and tables."""
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(A4),
        rightMargin=12 * mm, leftMargin=12 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
        title="Machine Age Intelligence Report", author="Machine Age Intelligence"
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitleCustom", parent=styles["Title"], textColor=colors.HexColor("#0B1F3A"),
        alignment=TA_CENTER, fontSize=22, leading=26, spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name="SectionHeadingCustom", parent=styles["Heading2"], textColor=colors.HexColor("#1565C0"),
        fontSize=13, leading=16, spaceBefore=8, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="SmallNoteCustom", parent=styles["BodyText"], fontSize=8, leading=10,
        textColor=colors.HexColor("#667085")
    ))
    story = []
    qty_total = float(filtered["Quantity"].sum()) if not filtered.empty else 0
    avg_age = weighted_average(filtered)
    age16 = float(filtered.loc[filtered["Age Years"] >= 16, "Quantity"].sum()) if not filtered.empty else 0
    missing_age = float(filtered.loc[filtered["Age Years"].isna(), "Quantity"].sum()) if not filtered.empty else 0
    unit_count = filtered["Unit"].nunique() if not filtered.empty else 0
    category_count = filtered["Category"].nunique() if not filtered.empty else 0

    story.append(Paragraph("Machine Age Intelligence", styles["ReportTitleCustom"]))
    story.append(Paragraph("Kenpark &amp; Regency — Machine Master Data Age Analysis", styles["Heading2"]))
    story.append(Paragraph(
        f"Age as of: <b>{pd.Timestamp(report_date).strftime('%d %b %Y')}</b> &nbsp; | &nbsp; "
        f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')} &nbsp; | &nbsp; "
        f"Filtered rows: {len(filtered):,}", styles["SmallNoteCustom"]
    ))
    story.append(Spacer(1, 6 * mm))

    kpi_rows = [
        ["Total machines", "Weighted average age", "Machines aged 16+ years", "Running units", "Categories"],
        [f"{qty_total:,.0f}", "—" if pd.isna(avg_age) else f"{avg_age:.1f} years",
         f"{age16:,.0f} ({age16 / qty_total:.1%})" if qty_total else "0",
         f"{unit_count:,}", f"{category_count:,}"],
        ["Age unavailable", "Age 16+ share", "Selected units", "Selected categories", "Selected age bands"],
        [f"{missing_age:,.0f}", f"{age16 / qty_total:.1%}" if qty_total else "0.0%",
         str(len(selected_units)), str(len(selected_cats)), str(len(age_filter))]
    ]
    kpi_table = Table(kpi_rows, colWidths=[(landscape(A4)[0] - 24 * mm) / 5] * 5)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F3A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EAF2FF")),
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 2), (-1, 2), colors.white),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F4F7FB")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#D9E2F0")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(Paragraph("1. Executive Summary", styles["SectionHeadingCustom"]))
    story.append(kpi_table)
    story.append(Spacer(1, 4 * mm))
    filter_text = (
        "Units: " + (", ".join(map(str, selected_units)) if selected_units else "None")
        + "<br/>Categories: " + (", ".join(map(str, selected_cats)) if selected_cats else "None")
        + "<br/>Age bands: " + (", ".join(map(str, age_filter)) if age_filter else "None")
        + "<br/>Search: " + (search.strip() if search and search.strip() else "None")
    )
    story.append(Paragraph(filter_text, styles["SmallNoteCustom"]))

    # High-resolution PDF charts with consistent, print-friendly styling.
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.labelsize": 10, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1", "grid.color": "#E2E8F0", "grid.linewidth": 0.7,
    })
    chart_buffers = []
    if not exact.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(exact["Age Whole Years"].astype(str), exact["Machine Qty"], color="#1565C0", edgecolor="white")
        ax.set_title("Exact Age Distribution")
        ax.set_xlabel("Age in whole years")
        ax.set_ylabel("Machine quantity")
        ax.tick_params(axis="x", labelrotation=60)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        chart_buffers.append(("Exact Age Distribution", buf))

        if "Cumulative %" in exact.columns:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(exact["Age Whole Years"], exact["Cumulative %"] * 100, marker="o", color="#F59E0B", linewidth=2.5, markersize=4)
            ax.set_title("Cumulative Machine Quantity by Age")
            ax.set_xlabel("Age in whole years")
            ax.set_ylabel("Cumulative share (%)")
            ax.set_ylim(0, 105)
            ax.grid(alpha=.25)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            buf.seek(0)
            chart_buffers.append(("Cumulative Age Profile", buf))

    if not overall_bands.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        bands = overall_bands.copy()
        ax.barh(bands["Age Band"].astype(str), bands["Machine Qty"], color=[AGE_COLORS.get(str(v), "#1565C0") for v in bands["Age Band"]])
        ax.set_title("Machine Quantity by Age Band")
        ax.set_xlabel("Machine quantity")
        ax.grid(axis="x", alpha=.25)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        chart_buffers.append(("Age Bands", buf))

    if not unit_summary.empty:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(unit_summary))
        bottom = np.zeros(len(unit_summary))
        for band in AGE_BANDS:
            values = pd.to_numeric(unit_summary[band], errors="coerce").fillna(0).to_numpy(dtype=float)
            ax.bar(x, values, bottom=bottom, label=band, color=AGE_COLORS[band])
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(unit_summary["Unit"].astype(str), rotation=45, ha="right")
        ax.set_title("Unit-wise Machine Age Composition")
        ax.set_ylabel("Machine quantity")
        ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, -0.2))
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        chart_buffers.append(("Unit Composition", buf))

    if chart_buffers:
        # One chart per landscape page prevents clipping and keeps labels readable.
        for chart_index, (chart_title, buf) in enumerate(chart_buffers, start=1):
            story.append(PageBreak())
            heading = "2. Charts" if chart_index == 1 else "2. Charts (continued)"
            story.append(Paragraph(heading, styles["SectionHeadingCustom"]))
            story.append(Paragraph(chart_title, styles["Heading3"]))
            story.append(Spacer(1, 2 * mm))
            buf.seek(0)
            image = RLImage(buf)
            max_w, max_h = 260 * mm, 145 * mm
            scale = min(max_w / image.imageWidth, max_h / image.imageHeight)
            image.drawWidth = image.imageWidth * scale
            image.drawHeight = image.imageHeight * scale
            image.hAlign = "CENTER"
            story.append(image)

    story.append(PageBreak())
    story.append(Paragraph("3. Overall Exact Age Distribution", styles["SectionHeadingCustom"]))
    exact_pdf = exact.copy()
    if "Cumulative %" in exact_pdf.columns:
        exact_pdf["Cumulative %"] = exact_pdf["Cumulative %"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "—"
        )
    story.append(pdf_table(exact_pdf, max_rows=70, font_size=8))

    story.append(PageBreak())
    story.append(Paragraph("4. Unit-wise Age Summary", styles["SectionHeadingCustom"]))
    story.append(pdf_table(unit_summary, max_rows=55, font_size=7))

    story.append(PageBreak())
    story.append(Paragraph("5. Unit / Category Age Summary", styles["SectionHeadingCustom"]))
    story.append(pdf_table(unit_category, max_rows=70, font_size=6))

    story.append(PageBreak())
    story.append(Paragraph("6. Data Quality Summary", styles["SectionHeadingCustom"]))
    story.append(pdf_table(quality, max_rows=30, font_size=8))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Records Requiring Review (first 100 records)", styles["Heading3"]))
    story.append(pdf_table(issues, max_rows=100, font_size=5.5))

    story.append(PageBreak())
    story.append(Paragraph("7. Filtered Machine Details", styles["SectionHeadingCustom"]))
    story.append(Paragraph(
        "The PDF includes up to the first 500 filtered machine rows. Download the Excel workbook for all filtered records.",
        styles["SmallNoteCustom"]
    ))
    detail = filtered.copy()
    if "Age Band" in detail.columns:
        detail["Age Band"] = detail["Age Band"].astype("object").fillna("Unclassified")
    preferred = [c for c in [
        "Unit", "Category", "Quantity", "Purchase Date Parsed", "Age Years",
        "Age Whole Years", "Age Band", "Data Status"
    ] if c in detail.columns]
    extras = [c for c in detail.columns if c not in preferred]
    detail = detail[preferred + extras[:max(0, 12 - len(preferred))]]
    story.append(pdf_table(detail, max_rows=500, font_size=5.5))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(12 * mm, 7 * mm, "Machine Age Intelligence | Kenpark & Regency")
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 7 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


with st.sidebar:
    st.markdown("### Data source")
    upload = st.file_uploader("Upload machine master Excel", type=["xlsx", "xlsm"])
    st.caption("Designed for 2027-Kenpark & Regency Age Analysis Full Details.xlsx. The app detects its main columns automatically.")
    if not upload:
        st.info("Upload the Full Details workbook to start the analysis.")
        st.stop()
    file_bytes = upload.getvalue()
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as exc:
        st.error(f"Could not read this Excel workbook: {exc}")
        st.stop()
    sheet = st.selectbox("Master-data sheet", xls.sheet_names, index=0)

raw = load_excel(file_bytes, sheet)
if raw.empty:
    st.warning("The selected worksheet contains no data rows.")
    st.stop()

cols = list(raw.columns)
unit_guess = find_col(cols, ["MC CURRENT SITE", "RUNNING UNIT", "CURRENT SITE", "UNIT", "DEPARTMENT"])
cat_guess = find_col(cols, ["Machine Description", "Machine Category", "CATEGORY", "MACHINE TYPE"])
date_guess = find_col(cols, ["Purchasing Date", "IN HOUSE YEAR", "PURCHASE DATE", "IN HOUSE DATE"])
age_guess = find_col(cols, ["MC AGE", "machine age", "AGE"])
qty_guess = find_col(cols, ["QTY", "QUANTITY"])


def idx_or_zero(value):
    return cols.index(value) if value in cols else 0


with st.sidebar:
    st.markdown("### Column mapping")
    unit_col = st.selectbox("Unit / current site", cols, idx_or_zero(unit_guess))
    category_col = st.selectbox("Machine category", cols, idx_or_zero(cat_guess))
    purchase_options = ["<Not available>"] + cols
    purchase_col = st.selectbox(
        "Purchasing date", purchase_options,
        purchase_options.index(date_guess) if date_guess in purchase_options else 0
    )
    age_options = ["<Not available>"] + cols
    supplied_age_col = st.selectbox(
        "Existing age (fallback)", age_options,
        age_options.index(age_guess) if age_guess in age_options else 0
    )
    qty_options = ["<Each row = 1 machine>"] + cols
    qty_col = st.selectbox(
        "Quantity", qty_options,
        qty_options.index(qty_guess) + 1 if qty_guess in cols else 0
    )
    report_date = st.date_input(
        "Age as of", value=date(2027, 9, 30),
        min_value=date(1990, 1, 1), max_value=date(2100, 12, 31)
    )

purchase_col = None if purchase_col == "<Not available>" else purchase_col
supplied_age_col = None if supplied_age_col == "<Not available>" else supplied_age_col
qty_col = None if qty_col == "<Each row = 1 machine>" else qty_col

if unit_col == category_col:
    st.error("Unit and category must use different columns.")
    st.stop()

df = prepare_data(raw, unit_col, category_col, purchase_col, supplied_age_col, qty_col, report_date)

with st.sidebar:
    st.markdown("### Filters")
    units = sorted(df["Unit"].dropna().unique().tolist())
    cats = sorted(df["Category"].dropna().unique().tolist())
    selected_units = st.multiselect("Units", units, default=units)
    selected_cats = st.multiselect("Categories", cats, default=cats)
    age_filter = st.multiselect("Age bands", AGE_BANDS, default=AGE_BANDS)
    search = st.text_input("Search asset / model / serial")

filtered = df[
    df["Unit"].isin(selected_units)
    & df["Category"].isin(selected_cats)
    & df["Age Band"].astype("object").fillna("Unclassified").isin(age_filter)
].copy()
if search.strip():
    mask = filtered.astype(str).apply(
        lambda col: col.str.contains(search.strip(), case=False, na=False, regex=False)
    ).any(axis=1)
    filtered = filtered[mask]

qty_total = float(filtered["Quantity"].sum())
avg_age = weighted_average(filtered)
age16 = float(filtered.loc[filtered["Age Years"] >= 16, "Quantity"].sum())
unclassified_qty = float(filtered.loc[filtered["Age Years"].isna(), "Quantity"].sum())
units_count = filtered["Unit"].nunique()
cats_count = filtered["Category"].nunique()

k1, k2, k3, k4, k5 = st.columns(5)


def kpi(container, label, value, hint):
    container.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="hint">{hint}</div></div>',
        unsafe_allow_html=True
    )


kpi(k1, "Total machines", f"{qty_total:,.0f}", "Filtered quantity")
kpi(k2, "Weighted avg age", "—" if np.isnan(avg_age) else f"{avg_age:.1f} yr", f"As of {report_date:%d %b %Y}")
kpi(k3, "Age 16+ years", f"{age16:,.0f}", f"{(age16 / qty_total * 100 if qty_total else 0):.1f}% of machines")
kpi(k4, "Running units", f"{units_count:,}", "Current-site groups")
kpi(k5, "Categories", f"{cats_count:,}", f"{unclassified_qty:,.0f} age-unclassified")

unit_summary = band_summary(filtered, ["Unit"])
unit_category = band_summary(filtered, ["Unit", "Category"])
exact = (
    filtered.dropna(subset=["Age Whole Years"])
    .groupby("Age Whole Years", observed=True)["Quantity"]
    .sum().rename("Machine Qty").reset_index().sort_values("Age Whole Years")
)
if not exact.empty:
    full_years = pd.DataFrame({"Age Whole Years": range(0, int(exact["Age Whole Years"].max()) + 1)})
    exact = full_years.merge(exact, on="Age Whole Years", how="left").fillna({"Machine Qty": 0})
    exact["Cumulative %"] = exact["Machine Qty"].cumsum() / max(exact["Machine Qty"].sum(), 1)
else:
    exact = pd.DataFrame(columns=["Age Whole Years", "Machine Qty", "Cumulative %"])

overall_bands = (
    filtered.groupby("Age Band", observed=False)["Quantity"].sum()
    .reindex(AGE_BANDS, fill_value=0).rename("Machine Qty").reset_index()
)
overall_bands["Share %"] = overall_bands["Machine Qty"] / max(overall_bands["Machine Qty"].sum(), 1)
quality = filtered.groupby("Data Status", dropna=False)["Quantity"].sum().rename("Machine Qty").reset_index()
issues = filtered[
    (filtered["Age Years"].isna())
    | (filtered["Age Years"] > 50)
    | (filtered["Unit"] == "Unclassified")
    | (filtered["Category"] == "Unclassified")
].copy()

st.markdown(
    f'<div class="status-strip"><span class="status-pill">Sheet: {sheet}</span>'
    f'<span class="status-pill">Rows: {len(filtered):,}</span>'
    f'<span class="status-pill">Units selected: {len(selected_units):,}</span>'
    f'<span class="status-pill">Categories selected: {len(selected_cats):,}</span>'
    f'<span class="status-pill">Age as of: {report_date:%d %b %Y}</span></div>',
    unsafe_allow_html=True
)
if filtered.empty:
    st.warning("No records match the current filters. Adjust the sidebar selections to continue.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🏭 Unit & Category", "📈 Visual Analysis", "✅ Data Quality", "🔎 Machine Details"
])

with tab1:
    st.markdown('<div class="section-title">Exact Age Distribution & Cumulative Profile</div>', unsafe_allow_html=True)
    if exact.empty:
        st.warning("No classified age records are available for the selected filters.")
    else:
        c1, c2 = st.columns([1.65, 1])
        fig = go.Figure()
        fig.add_bar(x=exact["Age Whole Years"], y=exact["Machine Qty"], name="Machine Qty", marker_color=BRAND["blue"])
        fig.add_scatter(
            x=exact["Age Whole Years"], y=exact["Cumulative %"], name="Cumulative %",
            yaxis="y2", line=dict(color=BRAND["orange"], width=3)
        )
        fig.update_layout(
            template="plotly_white", height=430, margin=dict(l=20, r=20, t=25, b=20),
            yaxis=dict(title="Machine Qty"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]),
            legend=dict(orientation="h", y=1.12)
        )
        c1.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(
            overall_bands, x="Machine Qty", y="Age Band", orientation="h", color="Age Band",
            color_discrete_map=AGE_COLORS, text_auto=".0f"
        )
        fig2.update_layout(
            template="plotly_white", showlegend=False, height=430, margin=dict(l=20, r=20, t=25, b=20),
            yaxis={"categoryorder": "array", "categoryarray": AGE_BANDS[::-1]}
        )
        c2.plotly_chart(fig2, use_container_width=True)
        st.dataframe(
            exact.style.format({"Machine Qty": "{:,.0f}", "Cumulative %": "{:.1%}"}),
            use_container_width=True, hide_index=True
        )

with tab2:
    st.markdown('<div class="section-title">Unit-Level Age Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        unit_summary.style.format({
            **{c: "{:,.0f}" for c in AGE_BANDS + ["Total Machines"]}, "Average Age": "{:.1f}"
        }).background_gradient(subset=[c for c in AGE_BANDS if c != "Unclassified"], cmap="YlOrRd"),
        use_container_width=True, hide_index=True
    )
    st.markdown('<div class="section-title">Unit / Category Detailed Age Summary</div>', unsafe_allow_html=True)
    st.dataframe(
        unit_category.style.format({
            **{c: "{:,.0f}" for c in AGE_BANDS + ["Total Machines"]}, "Average Age": "{:.1f}"
        }),
        use_container_width=True, hide_index=True, height=520
    )

with tab3:
    st.markdown('<div class="section-title">Unit-wise Machine Age Composition</div>', unsafe_allow_html=True)
    unit_long = unit_summary.melt(id_vars=["Unit"], value_vars=AGE_BANDS, var_name="Age Band", value_name="Machine Qty")
    fig = px.bar(
        unit_long, x="Unit", y="Machine Qty", color="Age Band",
        color_discrete_map=AGE_COLORS, barmode="stack", text_auto=False
    )
    fig.update_layout(
        template="plotly_white", height=440, margin=dict(l=20, r=20, t=15, b=80),
        legend=dict(orientation="h", y=1.15)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="section-title">Category-wise Machine Age Composition</div>', unsafe_allow_html=True)
    category_summary = band_summary(filtered, ["Category"]).sort_values("Total Machines", ascending=False)
    if len(category_summary) > 10:
        top_n = st.slider(
            "Number of categories in chart", min_value=10,
            max_value=min(60, len(category_summary)),
            value=min(25, len(category_summary))
        )
    else:
        top_n = len(category_summary)
    clong = category_summary.head(top_n).melt(
        id_vars=["Category"], value_vars=AGE_BANDS, var_name="Age Band", value_name="Machine Qty"
    )
    fig = px.bar(
        clong, x="Machine Qty", y="Category", color="Age Band",
        color_discrete_map=AGE_COLORS, barmode="stack"
    )
    fig.update_layout(
        template="plotly_white", height=max(460, top_n * 27), margin=dict(l=20, r=20, t=15, b=20),
        legend=dict(orientation="h", y=1.08), yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    a, b = st.columns([1, 1.4])
    a.markdown('<div class="section-title">Age Source Status</div>', unsafe_allow_html=True)
    fig = px.pie(
        quality, names="Data Status", values="Machine Qty", hole=.58,
        color_discrete_sequence=[BRAND["green"], BRAND["orange"], BRAND["red"]]
    )
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=10, b=10))
    a.plotly_chart(fig, use_container_width=True)
    b.markdown('<div class="section-title">Records Requiring Review</div>', unsafe_allow_html=True)
    b.caption("Shows missing age, ages above 50 years, or unclassified unit/category.")
    b.dataframe(issues, use_container_width=True, hide_index=True, height=330)

with tab5:
    st.markdown('<div class="section-title">Filtered Machine-Level Details</div>', unsafe_allow_html=True)
    preferred = [
        c for c in [
            unit_col, category_col, purchase_col, supplied_age_col, "Purchase Date Parsed",
            "Age Years", "Age Whole Years", "Age Band", "Quantity", "Data Status"
        ] if c and c in filtered.columns
    ]
    other = [c for c in filtered.columns if c not in preferred]
    detail_view = filtered[preferred + other].copy()
    if "Age Band" in detail_view.columns:
        detail_view["Age Band"] = detail_view["Age Band"].astype("object").fillna("Unclassified")
    st.dataframe(detail_view, use_container_width=True, hide_index=True, height=600)

# Exports: Excel workbook and PDF report both use the currently filtered data.
export_detail = filtered.copy()
export_detail["Age Band"] = export_detail["Age Band"].astype("object").fillna("Unclassified")
export = to_excel_bytes(export_detail, unit_summary, unit_category, exact, overall_bands, quality)

st.divider()
st.markdown("### Download reports")
st.caption("Both exports respect the current filters. The PDF uses dedicated landscape chart pages for clear printing.")
download_col1, download_col2 = st.columns(2)
with download_col1:
    st.download_button(
        "⬇️ Download full analysis workbook",
        data=export,
        file_name="Machine_Age_Analysis_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with download_col2:
    try:
        pdf_bytes = create_pdf_report(
            filtered=filtered,
            unit_summary=unit_summary,
            unit_category=unit_category,
            exact=exact,
            overall_bands=overall_bands,
            quality=quality,
            issues=issues,
            report_date=report_date,
            selected_units=selected_units,
            selected_cats=selected_cats,
            age_filter=age_filter,
            search=search
        )
        st.download_button(
            "📄 Download PDF report",
            data=pdf_bytes,
            file_name=f"Machine_Age_Analysis_{report_date:%Y%m%d}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as exc:
        st.error(f"Could not generate the PDF report: {exc}")
        st.caption("Check that reportlab and matplotlib are installed in your environment.")

st.caption(
    "Age is calculated from purchasing date first. Existing MC AGE is used only when the purchasing date cannot be parsed. "
    "All totals are quantity-weighted. PDF machine details are limited to 500 rows; use Excel for complete detail."
)
