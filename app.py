import html
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether,
)

st.set_page_config(page_title="Machine Age Intelligence", page_icon="🏭", layout="wide")

BRAND = {
    "navy": "#0B1F3A", "blue": "#1565C0", "cyan": "#00A6A6",
    "green": "#2E7D32", "orange": "#F59E0B", "red": "#C62828",
    "ink": "#172033", "muted": "#667085", "bg": "#F4F7FB", "card": "#FFFFFF",
}
AGE_BANDS = [
    "0–5 years", "6–10 years", "11–15 years", "16–20 years",
    "21–25 years", "26+ years", "Unclassified",
]
AGE_COLORS = {
    "0–5 years": "#2E7D32", "6–10 years": "#66BB6A",
    "11–15 years": "#F9A825", "16–20 years": "#EF6C00",
    "21–25 years": "#D84315", "26+ years": "#8E2430",
    "Unclassified": "#98A2B3",
}

# ---------------------------------------------------------------------------
# Streamlit version compatibility (use_container_width -> width="stretch")
# ---------------------------------------------------------------------------
try:
    _ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
except Exception:
    _ver = (1, 0)
STRETCH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
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
            if nc and (nc in norm(c) or (norm(c) and norm(c) in nc)):
                return c
    return None


def parse_date_value(v) -> pd.Timestamp:
    if v is None:
        return pd.NaT
    try:
        if pd.isna(v) or str(v).strip() == "":
            return pd.NaT
    except (TypeError, ValueError):
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
    if v is None or pd.isna(v):
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
    if v is None or pd.isna(v):
        return fallback
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s.upper() if s else fallback


@st.cache_data(show_spinner=False)
def load_table(file_bytes: bytes, file_name: str, sheet_name):
    if file_name.lower().endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(file_bytes), dtype=object, encoding_errors="replace")
    else:
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
    if purchase_col:
        parsed = df[purchase_col].map(parse_date_value)
        df["Purchase Date Parsed"] = pd.to_datetime(parsed, errors="coerce")
    else:
        df["Purchase Date Parsed"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    cutoff = pd.Timestamp(report_date)
    date_age = (cutoff - df["Purchase Date Parsed"]).dt.days / 365.2425
    if supplied_age_col:
        supplied_age = pd.to_numeric(df[supplied_age_col].map(age_from_text), errors="coerce")
    else:
        supplied_age = pd.Series(np.nan, index=df.index, dtype="float64")
    # Purchase date is preferred; supplied age is used only if the date is unavailable.
    df["Age Years"] = pd.to_numeric(date_age.where(date_age.notna(), supplied_age), errors="coerce")
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
        default="Age unavailable",
    )
    return df


def weighted_average(df):
    if df is None or df.empty:
        return np.nan
    good = df["Age Years"].notna() & (df["Quantity"] > 0)
    if not good.any():
        return np.nan
    return float(np.average(df.loc[good, "Age Years"], weights=df.loc[good, "Quantity"]))


def band_summary(df, group_cols):
    if df.empty:
        return pd.DataFrame(columns=group_cols + AGE_BANDS + ["Total Machines", "Average Age"])
    out = df.groupby(group_cols + ["Age Band"], observed=False)["Quantity"].sum().unstack(fill_value=0)
    out = out.reindex(columns=AGE_BANDS, fill_value=0).reset_index()
    out["Total Machines"] = out[AGE_BANDS].sum(axis=1)
    average_rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, group in df.groupby(grouper, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["Average Age"] = weighted_average(group)
        average_rows.append(row)
    averages = pd.DataFrame(average_rows, columns=group_cols + ["Average Age"])
    out = out.merge(averages, on=group_cols, how="left")
    sort_cols = group_cols[:-1] + ["Total Machines"] if len(group_cols) > 1 else ["Total Machines"]
    asc = [True] * (len(group_cols) - 1) + [False] if len(group_cols) > 1 else [False]
    return out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


def build_issues(filtered):
    """Records that need a human look, with the reason spelled out."""
    cols = ["Review Reason"] + list(filtered.columns)
    if filtered.empty:
        return pd.DataFrame(columns=cols)
    flags = pd.DataFrame({
        "Age unavailable": filtered["Age Years"].isna(),
        "Age above 50 years": filtered["Age Years"] > 50,
        "Unit missing": filtered["Unit"] == "Unclassified",
        "Category missing": filtered["Category"] == "Unclassified",
    })
    issues = filtered[flags.any(axis=1)].copy()
    if issues.empty:
        return pd.DataFrame(columns=cols)
    reasons = flags.loc[issues.index].apply(lambda r: "; ".join(k for k, v in r.items() if v), axis=1)
    issues.insert(0, "Review Reason", reasons)
    return issues


def compute_tables(filtered):
    unit_summary = band_summary(filtered, ["Unit"])
    unit_category = band_summary(filtered, ["Unit", "Category"])
    category_summary = band_summary(filtered, ["Category"])
    exact = (
        filtered.dropna(subset=["Age Whole Years"])
        .groupby("Age Whole Years", observed=True)["Quantity"]
        .sum().rename("Machine Qty").reset_index().sort_values("Age Whole Years")
    )
    if not exact.empty:
        exact["Age Whole Years"] = exact["Age Whole Years"].astype(int)
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
    quality["Share %"] = quality["Machine Qty"] / max(quality["Machine Qty"].sum(), 1)
    issues = build_issues(filtered)
    return dict(
        unit_summary=unit_summary, unit_category=unit_category, category_summary=category_summary,
        exact=exact, overall_bands=overall_bands, quality=quality, issues=issues,
    )


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
        header = wb.add_format({"bold": True, "font_color": "white", "bg_color": "#0B1F3A", "valign": "vcenter"})
        pct = wb.add_format({"num_format": "0.0%"})
        sheets = {
            "Filtered Detail": detail, "Unit Age Summary": unit_summary, "Unit Category Age": unit_category,
            "Overall Exact Age": overall_exact, "Overall Age Bands": overall_bands, "Data Quality": quality,
        }
        for name, frame in sheets.items():
            ws = writer.sheets[name]
            ws.freeze_panes(1, 0)
            ws.set_row(0, 22)
            for i, col in enumerate(frame.columns):
                ws.write(0, i, col, header)
                try:
                    longest = pd.Series(frame.iloc[:200, i]).map(lambda v: len(str(v)) if pd.notna(v) else 0).max()
                    longest = 0 if pd.isna(longest) else int(longest)
                except Exception:
                    longest = 12
                width = min(40, max(12, len(str(col)) + 2, longest + 2))
                fmt = pct if "%" in str(col) else None
                ws.set_column(i, i, width, fmt)
            if len(frame.columns):
                ws.autofilter(0, 0, max(1, len(frame)), len(frame.columns) - 1)
    return output.getvalue()


# ---------------------------------------------------------------------------
# PDF engine
# ---------------------------------------------------------------------------
PAGE_W, PAGE_H = landscape(A4)
MARGIN = 12 * mm
CONTENT_W = PAGE_W - 2 * MARGIN - 6 * mm   # leaves room for the frame padding
CONTENT_H = 142 * mm                       # max image height under a heading
C = lambda key: colors.HexColor(BRAND[key])  # noqa: E731


def _esc(text):
    return html.escape(str(text), quote=False)


def _short(text, n=32, ellipsis="…"):
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + ellipsis


def pdf_styles():
    base = getSampleStyleSheet()
    mk = lambda name, **kw: ParagraphStyle(name, parent=base["BodyText"], **kw)  # noqa: E731
    ink = C("ink")
    return {
        "h1": mk("h1", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=C("navy"), spaceAfter=3),
        "h2": mk("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=C("blue"), spaceBefore=8, spaceAfter=4),
        "body": mk("body", fontName="Helvetica", fontSize=9.5, leading=14, textColor=ink),
        "bullet": mk("bullet", fontName="Helvetica", fontSize=9, leading=12, textColor=ink, leftIndent=12, bulletIndent=2, spaceAfter=2),
        "note": mk("note", fontName="Helvetica-Oblique", fontSize=8, leading=10.5, textColor=C("muted")),
        "cover_title": mk("ct", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=colors.white),
        "cover_sub": mk("cs", fontName="Helvetica", fontSize=12, leading=16, textColor=colors.HexColor("#DBEAFE")),
        "cover_meta": mk("cm", fontName="Helvetica", fontSize=8.5, leading=12, textColor=colors.HexColor("#BFD6F5")),
        "kpi_label": mk("kl", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=C("muted")),
        "kpi_value": mk("kv", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=C("navy")),
        "kpi_hint": mk("kh", fontName="Helvetica", fontSize=7.5, leading=9.5, textColor=C("muted")),
    }


def make_numbered_canvas(report_title, org_name):
    class NumberedCanvas(rl_canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_pages = []

        def showPage(self):
            self._saved_pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_pages)
            for state in self._saved_pages:
                self.__dict__.update(state)
                self._draw_chrome(total)
                super().showPage()
            super().save()

        def _draw_chrome(self, total):
            page = self._pageNumber
            self.saveState()
            if page > 1:
                self.setStrokeColor(C("navy"))
                self.setLineWidth(1.2)
                self.line(MARGIN, PAGE_H - 10 * mm, PAGE_W - MARGIN, PAGE_H - 10 * mm)
                self.setFillColor(C("navy"))
                self.setFont("Helvetica-Bold", 8)
                self.drawString(MARGIN, PAGE_H - 8 * mm, report_title)
                if org_name:
                    self.setFont("Helvetica", 8)
                    self.drawRightString(PAGE_W - MARGIN, PAGE_H - 8 * mm, org_name)
            self.setStrokeColor(colors.HexColor("#D9E2F0"))
            self.setLineWidth(0.6)
            self.line(MARGIN, 11 * mm, PAGE_W - MARGIN, 11 * mm)
            self.setFillColor(C("muted"))
            self.setFont("Helvetica", 7.5)
            self.drawString(MARGIN, 7 * mm, f"{report_title}  |  Confidential – for internal management use")
            self.drawRightString(PAGE_W - MARGIN, 7 * mm, f"Page {page} of {total}")
            self.restoreState()

    return NumberedCanvas


def pdf_cell(value, col="", max_chars=30, zero_dash=False):
    """Plain-text formatting for table cells (escaped later)."""
    if value is None:
        return "–"
    try:
        if pd.isna(value):
            return "–"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).strftime("%d-%b-%Y")
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, np.integer)):
        if zero_dash and value == 0:
            return "–"
        return f"{value:,}" if abs(value) >= 10000 else str(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if zero_dash and v == 0:
            return "–"
        if v.is_integer():
            return f"{v:,.0f}" if abs(v) >= 10000 else f"{v:.0f}"
        return f"{v:,.1f}"
    return _short(value, max_chars, "...")


def pdf_table(df, styles, max_rows=60, font_size=7, max_width=None, zero_dash=False,
              bold_last=False, max_chars=30):
    """Wrapped-text table with content-aware column widths. Returns a list of flowables."""
    note = None
    if df is None or df.empty:
        view = pd.DataFrame({"Information": ["No records available"]})
    else:
        view = df.head(max_rows).copy()
        if len(df) > max_rows:
            note = f"Showing the first {max_rows:,} of {len(df):,} rows."
    cols = list(view.columns)
    is_num = [bool(pd.api.types.is_numeric_dtype(view[c])) for c in cols]
    body = [[pdf_cell(v, c, max_chars, zero_dash and is_num[i]) for i, (c, v) in enumerate(zip(cols, row))]
            for row in view.itertuples(index=False, name=None)]
    numeric = [is_num[i] or (len(body) > 0 and all(re.fullmatch(r"[\d,.\-–%]+", r[i]) for r in body))
               for i in range(len(cols))]

    head_style = ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=font_size, leading=font_size + 2,
                                textColor=colors.white, alignment=TA_CENTER)
    left = ParagraphStyle("tl", fontName="Helvetica", fontSize=font_size, leading=font_size + 2,
                          textColor=C("ink"), alignment=TA_LEFT)
    right = ParagraphStyle("tr", parent=left, alignment=TA_RIGHT)

    data = [[Paragraph(_esc(c), head_style) for c in cols]]
    for r_i, row in enumerate(body):
        is_total = bold_last and r_i == len(body) - 1
        cells = []
        for c_i, text in enumerate(row):
            txt = _esc(text)
            if is_total:
                txt = f"<b>{txt}</b>"
            cells.append(Paragraph(txt, right if numeric[c_i] else left))
        data.append(cells)

    weights = []
    for i, c in enumerate(cols):
        longest = max([len(str(c)) * 0.75] + [len(r[i]) for r in body[:200]])
        lo, hi = (6, 13) if numeric[i] else (8, 34)
        weights.append(min(max(longest, lo), hi))
    total_w = min(CONTENT_W, max_width) if max_width else CONTENT_W
    widths = [w / sum(weights) * total_w for w in weights]

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT" if max_width else "CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), C("navy")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FB")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if bold_last and len(data) > 2:
        style += [("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#DCE8F8")),
                  ("LINEABOVE", (0, -1), (-1, -1), 1, C("navy"))]
    table.setStyle(TableStyle(style))
    out = [table]
    if note:
        out += [Spacer(1, 1.5 * mm), Paragraph(note, styles["note"])]
    return out


# ---- charts ---------------------------------------------------------------
def _style_axes(*axes):
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#CBD5E1")
        ax.tick_params(colors="#344054", labelsize=8)
        ax.set_axisbelow(True)


def _save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def chart_age_profile(exact, overall_bands, thr):
    if exact.empty and overall_bands["Machine Qty"].sum() == 0:
        return None
    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.55)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _style_axes(ax, ax2)

    if exact.empty:
        ax.text(0.5, 0.5, "No machines with a usable age", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        x = exact["Age Whole Years"].astype(int).to_numpy()
        y = exact["Machine Qty"].to_numpy(float)
        ax.bar(x, y, color=[BRAND["blue"] if a < thr else BRAND["red"] for a in x], width=0.8)
        if len(x) <= 35:
            for xi, yi in zip(x, y):
                if yi > 0:
                    ax.text(xi, yi, f"{yi:,.0f}", ha="center", va="bottom", fontsize=7, color="#344054")
        ax.set_xticks(x[:: max(1, len(x) // 20)])
        ax.set_xlabel("Age in whole years", fontsize=9)
        ax.set_ylabel("Machine quantity", fontsize=9)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
        ax.set_title("Exact age distribution and cumulative share", fontsize=11, fontweight="bold", loc="left", color=BRAND["navy"])
        tw = ax.twinx()
        tw.plot(x, exact["Cumulative %"].to_numpy(float) * 100, color=BRAND["orange"], marker="o", markersize=3.5, linewidth=2.2)
        tw.set_ylim(0, 105)
        tw.set_ylabel("Cumulative share (%)", fontsize=9)
        tw.spines["top"].set_visible(False)
        tw.tick_params(labelsize=8, colors="#344054")
        ax.legend(handles=[
            Patch(color=BRAND["blue"], label=f"Under {thr} years"),
            Patch(color=BRAND["red"], label=f"{thr}+ years"),
            Line2D([0], [0], color=BRAND["orange"], marker="o", markersize=4, label="Cumulative %"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=8)

    b = overall_bands
    vals = b["Machine Qty"].to_numpy(float)
    ypos = np.arange(len(b))[::-1]
    ax2.barh(ypos, vals, color=[AGE_COLORS.get(str(k), BRAND["blue"]) for k in b["Age Band"]], height=0.62)
    total = vals.sum()
    for yp, v in zip(ypos, vals):
        label = f"{v:,.0f} ({v / total:.0%})" if total else "0"
        ax2.text(v + max(vals.max(), 1) * 0.02, yp, label, va="center", fontsize=8, color="#344054")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(b["Age Band"].astype(str), fontsize=8.5)
    ax2.set_xlim(0, max(vals.max() * 1.45, 1))
    ax2.set_xlabel("Machine quantity", fontsize=9)
    ax2.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax2.set_title("Quantity by age band", fontsize=11, fontweight="bold", loc="left", color=BRAND["navy"])
    return _save_fig(fig)


def old_share(filtered, col, thr):
    total = filtered.groupby(col, observed=True)["Quantity"].sum()
    old = filtered[filtered["Age Years"] >= thr].groupby(col, observed=True)["Quantity"].sum()
    return old.reindex(total.index).fillna(0) / total.replace(0, np.nan) * 100


def chart_composition(summary, label_col, metric, metric_title, metric_fmt, metric_ref=None,
                      top_n=20, left_title="Machine quantity by age band", color_by_age=False):
    """Stacked age-band bars (left) + one comparison metric (right) sharing the same rows."""
    if summary is None or summary.empty:
        return None
    data = summary.sort_values("Total Machines", ascending=False).head(top_n).iloc[::-1]
    n = len(data)
    fig_h = min(max(4.6, 0.38 * n + 2.0), 9.0)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, fig_h), sharey=True,
        gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.06},
    )
    _style_axes(ax, ax2)
    ypos = np.arange(n)
    left = np.zeros(n)
    handles = []
    for band in AGE_BANDS:
        vals = pd.to_numeric(data[band], errors="coerce").fillna(0).to_numpy(float)
        if vals.sum() <= 0:
            continue
        ax.barh(ypos, vals, left=left, color=AGE_COLORS[band], height=0.66, edgecolor="white", linewidth=0.4)
        handles.append(Patch(color=AGE_COLORS[band], label=band))
        left += vals
    xmax = max(left.max(), 1)
    for yp, tot in zip(ypos, left):
        ax.text(tot + xmax * 0.012, yp, f"{tot:,.0f}", va="center", fontsize=8, fontweight="bold", color=BRAND["navy"])
    ax.set_xlim(0, xmax * 1.12)
    ax.set_yticks(ypos)
    ax.set_yticklabels([_short(v, 34) for v in data[label_col]], fontsize=8.5)
    ax.set_xlabel("Machine quantity", fontsize=9)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax.set_title(left_title, fontsize=11, fontweight="bold", loc="left", color=BRAND["navy"])

    mvals = metric.reindex(data[label_col]).to_numpy(float)
    safe = np.nan_to_num(mvals, nan=0.0)
    cols = ([AGE_COLORS[age_band(v)] if np.isfinite(v) else AGE_COLORS["Unclassified"] for v in mvals]
            if color_by_age else [BRAND["blue"]] * n)
    ax2.barh(ypos, safe, color=cols, height=0.66)
    mmax = max(np.nanmax(safe) if n else 1, metric_ref or 0, 1)
    for yp, v in zip(ypos, mvals):
        ax2.text((0 if np.isnan(v) else v) + mmax * 0.02, yp, "n/a" if np.isnan(v) else metric_fmt(v),
                 va="center", fontsize=8, color="#344054")
    if metric_ref is not None and np.isfinite(metric_ref):
        ax2.axvline(metric_ref, color=BRAND["navy"], linestyle="--", linewidth=1)
        ax2.text(metric_ref, n - 0.35, f" overall {metric_fmt(metric_ref)}", fontsize=7.5, color=BRAND["navy"], va="bottom")
    ax2.set_xlim(0, mmax * 1.35)
    ax2.set_xlabel(metric_title, fontsize=9)
    ax2.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    ax2.tick_params(axis="y", left=False)
    ax2.set_title(metric_title, fontsize=11, fontweight="bold", loc="left", color=BRAND["navy"])
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), frameon=False, fontsize=8,
               bbox_to_anchor=(0.45, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return _save_fig(fig)


def fit_image(buf, max_w=CONTENT_W, max_h=CONTENT_H):
    buf.seek(0)
    img = RLImage(buf)
    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    img.hAlign = "CENTER"
    return img


# ---- narrative ------------------------------------------------------------
def build_insights(filtered, unit_summary, category_summary, thr, qty_total, avg_age, old_qty, missing_qty):
    out = []
    if qty_total <= 0:
        return ["No machines match the current filters."]
    avg_txt = "not available" if pd.isna(avg_age) else f"{avg_age:.1f} years"
    out.append(f"The selection covers <b>{qty_total:,.0f}</b> machines with a quantity-weighted average age of <b>{avg_txt}</b>.")
    out.append(f"<b>{old_qty:,.0f}</b> machines (<b>{old_qty / qty_total:.1%}</b>) are <b>{thr} years or older</b> and are the main candidates for replacement or closer maintenance review.")
    valid = unit_summary[(unit_summary["Total Machines"] > 0) & unit_summary["Average Age"].notna()] if not unit_summary.empty else unit_summary
    if len(valid) >= 2:
        oldest, youngest = valid.loc[valid["Average Age"].idxmax()], valid.loc[valid["Average Age"].idxmin()]
        out.append(f"<b>{_esc(oldest['Unit'])}</b> has the oldest fleet (average {oldest['Average Age']:.1f} years), while <b>{_esc(youngest['Unit'])}</b> has the youngest ({youngest['Average Age']:.1f} years).")
    old_rows = filtered[filtered["Age Years"] >= thr]
    if not old_rows.empty:
        by_unit = old_rows.groupby("Unit")["Quantity"].sum()
        by_cat = old_rows.groupby("Category")["Quantity"].sum()
        out.append(f"The largest number of {thr}+ year machines sits in <b>{_esc(by_unit.idxmax())}</b> ({by_unit.max():,.0f} machines); by category, <b>{_esc(_short(by_cat.idxmax(), 50, '...'))}</b> leads with {by_cat.max():,.0f}.")
    if missing_qty > 0:
        out.append(f"<b>{missing_qty:,.0f}</b> machines ({missing_qty / qty_total:.1%}) have no usable age and are excluded from age statistics; see the Data Quality section.")
    else:
        out.append("Every machine in the selection has a usable age, so the statistics above are fully covered.")
    return out


def _scope_text(selected, everything):
    selected = list(selected)
    if everything is not None and len(selected) == len(everything):
        return f"All ({len(selected):,})"
    if not selected:
        return "None"
    shown = ", ".join(_short(s, 28, "...") for s in selected[:6])
    return shown + (f"  (+{len(selected) - 6:,} more)" if len(selected) > 6 else "")


# ---- report ---------------------------------------------------------------
def create_pdf_report(filtered, unit_summary, unit_category, exact, overall_bands, quality,
                      issues, report_date, selected_units, selected_cats, age_filter, search,
                      report_title="Machine Age Intelligence", org_name="", old_threshold=16,
                      all_units=None, all_cats=None, extra_detail_cols=None, category_summary=None):
    """Create a filter-aware, management-ready PDF with charts and readable tables."""
    thr = int(old_threshold)
    S = pdf_styles()
    story = []
    if category_summary is None:
        category_summary = band_summary(filtered, ["Category"])

    qty_total = float(filtered["Quantity"].sum()) if not filtered.empty else 0.0
    avg_age = weighted_average(filtered)
    old_qty = float(filtered.loc[filtered["Age Years"] >= thr, "Quantity"].sum()) if not filtered.empty else 0.0
    missing_qty = float(filtered.loc[filtered["Age Years"].isna(), "Quantity"].sum()) if not filtered.empty else 0.0
    max_age = filtered["Age Years"].max() if not filtered.empty else np.nan
    unit_count = filtered["Unit"].nunique() if not filtered.empty else 0
    category_count = filtered["Category"].nunique() if not filtered.empty else 0
    coverage = (1 - missing_qty / qty_total) if qty_total else 0.0

    # ---- page 1: title block, KPIs, insights, scope ----
    meta = (f"Age calculated as of <b>{pd.Timestamp(report_date):%d %b %Y}</b> &nbsp;|&nbsp; "
            f"Generated {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; {len(filtered):,} records")
    title_block = Table(
        [[[Paragraph(_esc(report_title), S["cover_title"]),
           Paragraph(_esc(org_name) if org_name else "Machine age analysis by unit and category", S["cover_sub"]),
           Spacer(1, 2 * mm), Paragraph(meta, S["cover_meta"])]]],
        colWidths=[CONTENT_W],
    )
    title_block.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C("navy")),
        ("LINEBELOW", (0, 0), (-1, -1), 3, C("cyan")),
        ("LEFTPADDING", (0, 0), (-1, -1), 14), ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [title_block, Spacer(1, 3 * mm), Paragraph("Executive summary", S["h1"])]

    kpis = [
        ("TOTAL MACHINES", f"{qty_total:,.0f}", "Filtered quantity"),
        ("AVERAGE AGE", "–" if pd.isna(avg_age) else f"{avg_age:.1f} yrs", "Quantity-weighted"),
        (f"AGED {thr}+ YEARS", f"{old_qty:,.0f}", f"{old_qty / qty_total:.1%} of machines" if qty_total else "–"),
        ("OLDEST MACHINE", "–" if pd.isna(max_age) else f"{max_age:.0f} yrs", "Highest single age"),
        ("UNITS", f"{unit_count:,}", "Current-site groups"),
        ("CATEGORIES", f"{category_count:,}", "Machine types"),
        ("AGE UNAVAILABLE", f"{missing_qty:,.0f}", "Excluded from age stats"),
        ("DATA COVERAGE", f"{coverage:.1%}", "Machines with a usable age"),
    ]
    cards = [[[Paragraph(l, S["kpi_label"]), Paragraph(v, S["kpi_value"]), Paragraph(h, S["kpi_hint"])] for l, v, h in kpis[i:i + 4]]
             for i in (0, 4)]
    kpi_table = Table(cards, colWidths=[CONTENT_W / 4] * 4)
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF4FB")),
        ("INNERGRID", (0, 0), (-1, -1), 4, colors.white),
        ("BOX", (0, 0), (-1, -1), 4, colors.white),
        ("LINEABOVE", (0, 0), (-1, -1), 2, C("blue")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [kpi_table, Spacer(1, 2 * mm), Paragraph("Key insights", S["h2"])]
    for line in build_insights(filtered, unit_summary, category_summary, thr, qty_total, avg_age, old_qty, missing_qty):
        story.append(Paragraph(line, S["bullet"], bulletText="•"))

    story += [Spacer(1, 1 * mm), Paragraph("Report scope (filters applied)", S["h2"])]
    scope = [
        ["Units", _scope_text(selected_units, all_units)],
        ["Categories", _scope_text(selected_cats, all_cats)],
        ["Age bands", f"All ({len(age_filter)})" if len(age_filter) == len(AGE_BANDS) else ", ".join(map(str, age_filter)) or "None"],
        ["Search text", search.strip() if search and search.strip() else "None"],
    ]
    lab = ParagraphStyle("sl", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=C("navy"))
    val = ParagraphStyle("sv", fontName="Helvetica", fontSize=8.5, leading=11, textColor=C("ink"))
    scope_t = Table([[Paragraph(a, lab), Paragraph(_esc(b), val)] for a, b in scope], colWidths=[28 * mm, CONTENT_W - 28 * mm])
    scope_t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2F0")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F7FB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(scope_t)

    # ---- chart pages ----
    chart_pages = [
        ("1. Age profile", "Left: machines by exact age with the cumulative share line (red bars are at or above the age threshold). Right: quantity per age band.",
         lambda: chart_age_profile(exact, overall_bands, thr)),
        ("2. Unit comparison", "Left: machine quantity by age band for each unit (largest units first). Right: weighted average age per unit; the dashed line is the overall average.",
         lambda: chart_composition(unit_summary, "Unit", unit_summary.set_index("Unit")["Average Age"] if not unit_summary.empty else None,
                                   "Average age (years)", lambda v: f"{v:.1f}", metric_ref=None if pd.isna(avg_age) else avg_age,
                                   top_n=20, left_title="Machine quantity by age band", color_by_age=True)),
        ("3. Category comparison", f"Left: machine quantity by age band for the largest categories. Right: share of each category that is {thr} years or older.",
         lambda: chart_composition(category_summary, "Category", old_share(filtered, "Category", thr),
                                   f"Share aged {thr}+ years (%)", lambda v: f"{v:.0f}%",
                                   metric_ref=(old_qty / qty_total * 100) if qty_total else None, top_n=15)),
    ]
    for heading, caption, builder in chart_pages:
        story.append(PageBreak())
        story += [Paragraph(heading, S["h1"]), Paragraph(caption, S["note"]), Spacer(1, 2 * mm)]
        try:
            buf = builder()
        except Exception as exc:  # never let one chart break the whole report
            buf = None
            story.append(Paragraph(f"Chart could not be rendered: {_esc(exc)}", S["body"]))
        if buf is not None:
            story.append(fit_image(buf))
        elif not any(isinstance(f, Paragraph) and "could not" in f.text for f in story[-2:]):
            story.append(Paragraph("No data available for this chart with the current filters.", S["body"]))

    # ---- tables ----
    story += [PageBreak(), Paragraph("4. Age distribution tables", S["h1"]), Paragraph("Quantity by age band", S["h2"])]
    bands_pdf = overall_bands.copy()
    bands_pdf["Age Band"] = bands_pdf["Age Band"].astype(str)
    bands_pdf["Share %"] = bands_pdf["Share %"].map(lambda v: f"{v:.1%}")
    story += pdf_table(bands_pdf, S, max_rows=20, font_size=8.5, max_width=120 * mm)
    story += [Spacer(1, 4 * mm), Paragraph("Exact age (whole years)", S["h2"])]
    exact_pdf = exact.copy()
    if "Cumulative %" in exact_pdf.columns:
        exact_pdf["Cumulative %"] = exact_pdf["Cumulative %"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "–")
    story += pdf_table(exact_pdf, S, max_rows=80, font_size=8.5, max_width=120 * mm)

    story += [PageBreak(), Paragraph("5. Unit-wise age summary", S["h1"]),
              Paragraph("Machine quantity per age band. Average age is quantity-weighted.", S["note"]), Spacer(1, 2 * mm)]
    unit_pdf = unit_summary.copy()
    if not unit_pdf.empty:
        total_row = {"Unit": "TOTAL", **{b: unit_pdf[b].sum() for b in AGE_BANDS},
                     "Total Machines": unit_pdf["Total Machines"].sum(), "Average Age": avg_age}
        unit_pdf = pd.concat([unit_pdf, pd.DataFrame([total_row])], ignore_index=True)
    story += pdf_table(unit_pdf, S, max_rows=80, font_size=8, zero_dash=True, bold_last=True)

    story += [PageBreak(), Paragraph("6. Unit and category age summary", S["h1"]),
              Paragraph("Grouped by unit, largest category first.", S["note"]), Spacer(1, 2 * mm)]
    story += pdf_table(unit_category, S, max_rows=250, font_size=7, zero_dash=True, max_chars=36)

    story += [PageBreak(), Paragraph("7. Data quality", S["h1"])]
    q_pdf = quality.copy()
    if "Share %" in q_pdf.columns:
        q_pdf["Share %"] = q_pdf["Share %"].map(lambda v: f"{v:.1%}")
    story += pdf_table(q_pdf, S, max_rows=20, font_size=8.5, max_width=150 * mm)
    story += [Spacer(1, 4 * mm), Paragraph("Records requiring review", S["h2"]),
              Paragraph(f"{len(issues):,} record(s) flagged for a missing age, an age above 50 years, or a missing unit/category.", S["note"]),
              Spacer(1, 2 * mm)]
    extras = [c for c in (extra_detail_cols or []) if c in issues.columns][:2]
    issue_cols = [c for c in ["Review Reason"] + extras + ["Unit", "Category", "Quantity", "Age Years", "Data Status"] if c in issues.columns]
    story += pdf_table(issues[issue_cols] if not issues.empty else issues, S, max_rows=100, font_size=7)

    story += [PageBreak(), Paragraph("8. Machine details", S["h1"]),
              Paragraph("Oldest machines first. The PDF lists up to 500 rows; download the Excel workbook for every filtered record.", S["note"]),
              Spacer(1, 2 * mm)]
    detail = filtered.copy()
    detail["Age Band"] = detail["Age Band"].astype("object").fillna("Unclassified")
    detail = detail.sort_values("Age Years", ascending=False, na_position="last").rename(columns={"Purchase Date Parsed": "Purchase Date"})
    extras = [c for c in (extra_detail_cols or []) if c in detail.columns][:3]
    wanted = extras + [c for c in ["Unit", "Category", "Quantity", "Purchase Date", "Age Years", "Age Band", "Data Status"] if c in detail.columns]
    story += pdf_table(detail[wanted], S, max_rows=500, font_size=6.5)

    story += [Spacer(1, 6 * mm), KeepTogether([
        Paragraph("Method notes", S["h2"]),
        Paragraph("Age is calculated from the purchase date up to the report date; a supplied age column is used only when no usable purchase date exists. "
                  "All totals and averages are weighted by machine quantity. Age bands: 0–5, 6–10, 11–15, 16–20, 21–25 and 26+ years.", S["note"]),
    ])]

    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(A4), leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16 * mm, bottomMargin=15 * mm, title=report_title, author=org_name or report_title,
    )
    doc.build(story, canvasmaker=make_numbered_canvas(report_title, org_name))
    return output.getvalue()


# ===================== STREAMLIT APP =====================
def inject_css():
    st.markdown(f"""
<style>
.stApp {{background: linear-gradient(180deg,#eef4fb 0,#f8fafc 230px); color:{BRAND['ink']};}}
.block-container {{padding-top:1.15rem; padding-bottom:3rem; max-width:1500px;}}
.hero {{padding:1.3rem 1.5rem; border-radius:20px; background:linear-gradient(120deg,{BRAND['navy']},{BRAND['blue']}); color:white; box-shadow:0 14px 38px rgba(11,31,58,.18); margin-bottom:1rem;}}
.hero h1 {{margin:0;font-size:2rem;color:white}} .hero p {{margin:.35rem 0 0;color:#dbeafe}}
.kpi {{background:white;border:1px solid #e6ebf2;border-radius:16px;padding:1rem 1.1rem;box-shadow:0 5px 18px rgba(16,24,40,.06);min-height:112px}}
.kpi .label {{color:{BRAND['muted']};font-size:.78rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}}
.kpi .value {{font-size:1.75rem;font-weight:800;color:{BRAND['navy']};margin-top:.3rem}}
.kpi .hint {{font-size:.78rem;color:{BRAND['muted']};margin-top:.2rem}}
.step {{background:white;border:1px solid #e6ebf2;border-radius:16px;padding:1.1rem 1.2rem;box-shadow:0 5px 18px rgba(16,24,40,.06);height:100%}}
.step .num {{display:inline-block;width:30px;height:30px;line-height:30px;text-align:center;border-radius:50%;background:{BRAND['blue']};color:white;font-weight:800;margin-bottom:.5rem}}
.step h4 {{margin:.1rem 0 .3rem;color:{BRAND['navy']}}} .step p {{margin:0;color:{BRAND['muted']};font-size:.9rem}}
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
""", unsafe_allow_html=True)


def show_df(frame, **kwargs):
    st.dataframe(frame, hide_index=True, **STRETCH, **kwargs)


def show_plot(fig):
    st.plotly_chart(fig, **STRETCH)


def kpi(container, label, value, hint):
    container.markdown(
        f'<div class="kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value}</div><div class="hint">{html.escape(hint)}</div></div>',
        unsafe_allow_html=True,
    )


def welcome():
    st.markdown('<div class="section-title">Get started in three steps</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><div class="num">1</div><h4>Upload your file</h4>'
                '<p>Use the sidebar to upload an Excel (.xlsx) or CSV file with one row per machine (or one row per machine type with a quantity).</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="step"><div class="num">2</div><h4>Check the columns</h4>'
                '<p>Unit, category, purchase date and quantity columns are detected automatically. Adjust them in the sidebar if needed.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="step"><div class="num">3</div><h4>Explore and export</h4>'
                '<p>Filter by unit, category or age band, then download a manager-ready PDF or a full Excel workbook.</p></div>', unsafe_allow_html=True)
    st.info("Tip: a purchase date (or year) gives the most accurate ages. An existing age column is used only as a fallback.")


inject_css()

with st.sidebar:
    st.markdown("### 📁 Data")
    upload = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xlsm", "csv"])
    st.markdown("### ⚙️ Report settings")
    report_title = st.text_input("Report title", "Machine Age Intelligence")
    org_name = st.text_input("Organisation / site (optional)", "", placeholder="e.g. Company name")
    report_date = st.date_input("Calculate age as of", value=date.today(),
                                min_value=date(1990, 1, 1), max_value=date(2100, 12, 31))
    old_threshold = int(st.number_input("Flag machines aged at least (years)", min_value=1, max_value=60, value=16, step=1))

st.markdown(
    f'<div class="hero"><h1>🏭 {html.escape(report_title or "Machine Age Intelligence")}</h1>'
    f'<p>{html.escape(org_name) + " · " if org_name else ""}Unit-wise and category-wise machine age analysis</p></div>',
    unsafe_allow_html=True,
)

if not upload:
    welcome()
    st.stop()

file_bytes = upload.getvalue()
is_csv = upload.name.lower().endswith(".csv")
sheet = None
if not is_csv:
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
    except Exception as exc:
        st.error(f"Could not read this Excel workbook: {exc}")
        st.stop()
    with st.sidebar:
        sheet = st.selectbox("Worksheet", xls.sheet_names, index=0)

try:
    raw = load_table(file_bytes, upload.name, sheet)
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()
if raw.empty:
    st.warning("The selected sheet contains no data rows.")
    st.stop()

cols = list(raw.columns)
unit_guess = find_col(cols, ["MC CURRENT SITE", "RUNNING UNIT", "CURRENT SITE", "UNIT", "DEPARTMENT", "LOCATION", "PLANT", "SITE"])
cat_guess = find_col(cols, ["Machine Description", "Machine Category", "CATEGORY", "MACHINE TYPE", "ASSET TYPE", "EQUIPMENT", "DESCRIPTION"])
date_guess = find_col(cols, ["Purchasing Date", "IN HOUSE YEAR", "PURCHASE DATE", "IN HOUSE DATE", "INSTALLATION DATE", "YEAR OF PURCHASE", "COMMISSION"])
age_guess = find_col(cols, ["MC AGE", "machine age", "AGE"])
qty_guess = find_col(cols, ["QTY", "QUANTITY"])


def idx_or_zero(value):
    return cols.index(value) if value in cols else 0


with st.sidebar:
    with st.expander("🧭 Column mapping (auto-detected)", expanded=not (unit_guess and cat_guess and (date_guess or age_guess))):
        unit_col = st.selectbox("Unit / current site", cols, idx_or_zero(unit_guess))
        category_col = st.selectbox("Machine category", cols, idx_or_zero(cat_guess))
        purchase_options = ["<Not available>"] + cols
        purchase_col = st.selectbox("Purchase date or year", purchase_options,
                                    purchase_options.index(date_guess) if date_guess in purchase_options else 0)
        age_options = ["<Not available>"] + cols
        supplied_age_col = st.selectbox("Existing age (fallback)", age_options,
                                        age_options.index(age_guess) if age_guess in age_options else 0)
        qty_options = ["<Each row = 1 machine>"] + cols
        qty_col = st.selectbox("Quantity", qty_options,
                               qty_options.index(qty_guess) + 1 if qty_guess in cols else 0)

purchase_col = None if purchase_col == "<Not available>" else purchase_col
supplied_age_col = None if supplied_age_col == "<Not available>" else supplied_age_col
qty_col = None if qty_col == "<Each row = 1 machine>" else qty_col

if unit_col == category_col:
    st.error("Unit and category must use different columns. Please change the column mapping in the sidebar.")
    st.stop()
if not purchase_col and not supplied_age_col:
    st.error("Select a purchase-date column or an existing-age column in the sidebar so ages can be calculated.")
    st.stop()

df = prepare_data(raw, unit_col, category_col, purchase_col, supplied_age_col, qty_col, report_date)

with st.sidebar:
    with st.expander("🔍 Filters", expanded=True):
        units = sorted(df["Unit"].dropna().unique().tolist())
        cats = sorted(df["Category"].dropna().unique().tolist())
        selected_units = st.multiselect("Units", units, default=units)
        selected_cats = st.multiselect("Categories", cats, default=cats)
        age_filter = st.multiselect("Age bands", AGE_BANDS, default=AGE_BANDS)
        search = st.text_input("Search any text", placeholder="asset / model / serial")

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

st.markdown(
    f'<div class="status-strip"><span class="status-pill">File: {html.escape(upload.name)}</span>'
    + (f'<span class="status-pill">Sheet: {html.escape(str(sheet))}</span>' if sheet else "")
    + f'<span class="status-pill">Rows: {len(filtered):,} of {len(df):,}</span>'
    f'<span class="status-pill">Units: {len(selected_units):,}</span>'
    f'<span class="status-pill">Categories: {len(selected_cats):,}</span>'
    f'<span class="status-pill">Age as of: {report_date:%d %b %Y}</span></div>',
    unsafe_allow_html=True,
)
if filtered.empty:
    st.warning("No records match the current filters. Adjust the sidebar selections to continue.")
    st.stop()

qty_total = float(filtered["Quantity"].sum())
avg_age = weighted_average(filtered)
old_qty = float(filtered.loc[filtered["Age Years"] >= old_threshold, "Quantity"].sum())
unclassified_qty = float(filtered.loc[filtered["Age Years"].isna(), "Quantity"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Total machines", f"{qty_total:,.0f}", "Filtered quantity")
kpi(k2, "Weighted avg age", "—" if np.isnan(avg_age) else f"{avg_age:.1f} yr", f"As of {report_date:%d %b %Y}")
kpi(k3, f"Age {old_threshold}+ years", f"{old_qty:,.0f}", f"{(old_qty / qty_total * 100 if qty_total else 0):.1f}% of machines")
kpi(k4, "Units", f"{filtered['Unit'].nunique():,}", "Current-site groups")
kpi(k5, "Categories", f"{filtered['Category'].nunique():,}", f"{unclassified_qty:,.0f} age-unclassified")
st.write("")

T = compute_tables(filtered)
unit_summary, unit_category, category_summary = T["unit_summary"], T["unit_category"], T["category_summary"]
exact, overall_bands, quality, issues = T["exact"], T["overall_bands"], T["quality"], T["issues"]

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
        fig.add_scatter(x=exact["Age Whole Years"], y=exact["Cumulative %"], name="Cumulative %",
                        yaxis="y2", line=dict(color=BRAND["orange"], width=3))
        fig.update_layout(
            template="plotly_white", height=430, margin=dict(l=20, r=20, t=25, b=20),
            xaxis=dict(title="Age (whole years)"), yaxis=dict(title="Machine Qty"),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]),
            legend=dict(orientation="h", y=1.12),
        )
        with c1:
            show_plot(fig)
        fig2 = px.bar(overall_bands, x="Machine Qty", y="Age Band", orientation="h", color="Age Band",
                      color_discrete_map=AGE_COLORS, text_auto=".0f")
        fig2.update_layout(template="plotly_white", showlegend=False, height=430, margin=dict(l=20, r=20, t=25, b=20),
                           yaxis={"categoryorder": "array", "categoryarray": AGE_BANDS[::-1]})
        with c2:
            show_plot(fig2)
        show_df(exact.style.format({"Machine Qty": "{:,.0f}", "Cumulative %": "{:.1%}"}))

with tab2:
    st.markdown('<div class="section-title">Unit-Level Age Summary</div>', unsafe_allow_html=True)
    show_df(
        unit_summary.style.format({**{c: "{:,.0f}" for c in AGE_BANDS + ["Total Machines"]}, "Average Age": "{:.1f}"}, na_rep="—")
        .background_gradient(subset=[c for c in AGE_BANDS if c != "Unclassified"], cmap="YlOrRd")
    )
    st.markdown('<div class="section-title">Unit / Category Detailed Age Summary</div>', unsafe_allow_html=True)
    show_df(
        unit_category.style.format({**{c: "{:,.0f}" for c in AGE_BANDS + ["Total Machines"]}, "Average Age": "{:.1f}"}, na_rep="—"),
        height=520,
    )

with tab3:
    st.markdown('<div class="section-title">Unit-wise Machine Age Composition</div>', unsafe_allow_html=True)
    unit_long = unit_summary.melt(id_vars=["Unit"], value_vars=AGE_BANDS, var_name="Age Band", value_name="Machine Qty")
    fig = px.bar(unit_long, x="Unit", y="Machine Qty", color="Age Band", color_discrete_map=AGE_COLORS, barmode="stack",
                 category_orders={"Age Band": AGE_BANDS})
    fig.update_layout(template="plotly_white", height=440, margin=dict(l=20, r=20, t=15, b=80), legend=dict(orientation="h", y=1.15))
    show_plot(fig)
    st.markdown('<div class="section-title">Category-wise Machine Age Composition</div>', unsafe_allow_html=True)
    cat_sorted = category_summary.sort_values("Total Machines", ascending=False)
    if len(cat_sorted) > 10:
        top_n = st.slider("Number of categories in chart", min_value=10, max_value=min(60, len(cat_sorted)),
                          value=min(25, len(cat_sorted)))
    else:
        top_n = len(cat_sorted)
    clong = cat_sorted.head(top_n).melt(id_vars=["Category"], value_vars=AGE_BANDS, var_name="Age Band", value_name="Machine Qty")
    fig = px.bar(clong, x="Machine Qty", y="Category", color="Age Band", color_discrete_map=AGE_COLORS, barmode="stack",
                 category_orders={"Age Band": AGE_BANDS})
    fig.update_layout(template="plotly_white", height=max(460, top_n * 27), margin=dict(l=20, r=20, t=15, b=20),
                      legend=dict(orientation="h", y=1.08), yaxis={"categoryorder": "total ascending"})
    show_plot(fig)

with tab4:
    a, b = st.columns([1, 1.4])
    a.markdown('<div class="section-title">Age Source Status</div>', unsafe_allow_html=True)
    fig = px.pie(quality, names="Data Status", values="Machine Qty", hole=.58,
                 color="Data Status",
                 color_discrete_map={"Calculated from purchase date": BRAND["green"], "Used supplied age": BRAND["orange"], "Age unavailable": BRAND["red"]})
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=10, r=10, t=10, b=10))
    with a:
        show_plot(fig)
    b.markdown('<div class="section-title">Records Requiring Review</div>', unsafe_allow_html=True)
    b.caption(f"{len(issues):,} record(s): missing age, age above 50 years, or unclassified unit/category.")
    with b:
        show_df(issues, height=330)

mapped_cols = {unit_col, category_col, purchase_col, supplied_age_col, qty_col}
derived_cols = {"Unit", "Category", "Quantity", "Purchase Date Parsed", "Age Years", "Age Whole Years", "Age Band", "Data Status"}
extra_detail_cols = [c for c in raw.columns if c not in mapped_cols and c not in derived_cols and df[c].notna().mean() > 0.5][:3]

with tab5:
    st.markdown('<div class="section-title">Filtered Machine-Level Details</div>', unsafe_allow_html=True)
    preferred = [c for c in [unit_col, category_col, purchase_col, supplied_age_col, "Purchase Date Parsed",
                             "Age Years", "Age Whole Years", "Age Band", "Quantity", "Data Status"]
                 if c and c in filtered.columns]
    preferred = list(dict.fromkeys(preferred))
    other = [c for c in filtered.columns if c not in preferred]
    detail_view = filtered[preferred + other].copy()
    detail_view["Age Band"] = detail_view["Age Band"].astype("object").fillna("Unclassified")
    show_df(detail_view, height=600)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
export_detail = filtered.copy()
export_detail["Age Band"] = export_detail["Age Band"].astype("object").fillna("Unclassified")

st.divider()
st.markdown("### 📥 Download reports")
st.caption("Both exports respect the current filters. Excel contains every record; the PDF is a print-ready summary for management.")
d1, d2 = st.columns(2)

with d1:
    try:
        excel_bytes = to_excel_bytes(export_detail, unit_summary, unit_category, exact, overall_bands, quality)
        st.download_button(
            "⬇️ Download Excel workbook", data=excel_bytes, file_name="Machine_Age_Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **STRETCH,
        )
    except Exception as exc:
        st.error(f"Could not generate the Excel workbook: {exc}")

with d2:
    signature = hash((
        upload.name, len(file_bytes), str(sheet), unit_col, category_col, purchase_col, supplied_age_col, qty_col,
        str(report_date), old_threshold, tuple(selected_units), tuple(selected_cats), tuple(age_filter),
        search, report_title, org_name,
    ))
    if st.button("📄 Prepare PDF report", **STRETCH):
        with st.spinner("Building PDF report…"):
            try:
                st.session_state["pdf_report"] = (signature, create_pdf_report(
                    filtered=filtered, unit_summary=unit_summary, unit_category=unit_category, exact=exact,
                    overall_bands=overall_bands, quality=quality, issues=issues, report_date=report_date,
                    selected_units=selected_units, selected_cats=selected_cats, age_filter=age_filter, search=search,
                    report_title=report_title or "Machine Age Intelligence", org_name=org_name,
                    old_threshold=old_threshold, all_units=units, all_cats=cats,
                    extra_detail_cols=extra_detail_cols, category_summary=category_summary,
                ))
            except Exception as exc:
                st.session_state.pop("pdf_report", None)
                st.error(f"Could not generate the PDF report: {exc}")
    saved = st.session_state.get("pdf_report")
    if saved and saved[0] == signature:
        st.download_button(
            "⬇️ Download PDF report", data=saved[1],
            file_name=f"Machine_Age_Analysis_{report_date:%Y%m%d}.pdf", mime="application/pdf", **STRETCH,
        )
    elif saved:
        st.caption("Filters or settings changed since the last PDF – click “Prepare PDF report” again to refresh it.")

st.caption(
    "Age is calculated from the purchase date first; an existing age column is used only when the date cannot be read. "
    "All totals are quantity-weighted. PDF machine details are limited to 500 rows; use Excel for complete detail."
)
