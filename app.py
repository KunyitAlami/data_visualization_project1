import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="UAS VisDat — Pernikahan, Perceraian, dan Affair",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Poppins', sans-serif !important; }
.block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }
h1 { font-weight: 700; }
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.96);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
}
section[data-testid="stSidebar"] {
  background: #FBFBFD;
  border-right: 1px solid rgba(0,0,0,0.06);
}
button[data-baseweb="tab"] { font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

GREEN = "#10B981"
GREEN_DARK = "#047857"
RED_PALETTE = ["#EF4444", "#F97316", "#E11D48", "#DC2626"]


def pick_red(i: int = 0) -> str:
    return RED_PALETTE[i % len(RED_PALETTE)]


def base_plotly_layout():
    return dict(
        template="plotly_white",
        font=dict(family="Poppins, Segoe UI, Arial", size=14, color="#111827"),
        margin=dict(l=18, r=18, t=90, b=70),  # top & bottom lebih lega
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,          # pindah ke bawah chart
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
    )



def apply_support_style(fig: go.Figure, supports_argument: bool, red_idx: int = 0) -> go.Figure:
    title_color = GREEN if supports_argument else pick_red(red_idx)
    fig.update_layout(base_plotly_layout())
    fig.update_layout(title=dict(x=0.01, xanchor="left", font=dict(size=18, color=title_color)))
    return fig

import plotly.io as pio

def download_plotly(fig: go.Figure, filename_prefix: str):
    """
    Download as PNG (kalau kaleido tersedia), fallback ke HTML.
    """
    c1, c2 = st.columns([1, 1])
    with c1:
        try:
            img_bytes = pio.to_image(fig, format="png", scale=2)
            st.download_button(
                label="⬇️ Download PNG",
                data=img_bytes,
                file_name=f"{filename_prefix}.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.download_button(
                label="⬇️ Download HTML (fallback)",
                data=fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8"),
                file_name=f"{filename_prefix}.html",
                mime="text/html",
                use_container_width=True,
            )
    with c2:
        st.download_button(
            label="⬇️ Download JSON",
            data=fig.to_json().encode("utf-8"),
            file_name=f"{filename_prefix}.json",
            mime="application/json",
            use_container_width=True,
        )


DATA_DIR = Path(__file__).parent / "data"


def _extract_year_from_name(name: str) -> int | None:
    m = re.search(r"(20\d{2})", str(name))
    return int(m.group(1)) if m else None


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]
    return df


# def _find_header_row(excel_path: Path, required_keywords: list[str], max_scan_rows: int = 25) -> int:
#     preview = pd.read_excel(excel_path, header=None, nrows=max_scan_rows)
#     best_row, best_score = 0, -1
#     for r in range(len(preview)):
#         row_vals = preview.iloc[r].astype(str).str.lower().str.strip().tolist()
#         score = sum(any(kw in v for v in row_vals) for kw in required_keywords)
#         if score > best_score:
#             best_row, best_score = r, score
#     return best_row

def _find_header_row(excel_path: Path, required_keywords: list[str], max_scan_rows: int = 25) -> int:
    preview = pd.read_excel(excel_path, header=None, nrows=max_scan_rows)

    required_keywords = [
        str(kw).lower().strip()
        for kw in required_keywords
        if pd.notna(kw)
    ]

    best_row, best_score = 0, -1

    for r in range(len(preview)):
        row_vals = []
        for v in preview.iloc[r].tolist():
            if pd.notna(v):
                row_vals.append(str(v).lower().strip())

        score = sum(
            any(kw in cell for cell in row_vals)
            for kw in required_keywords
        )

        if score > best_score:
            best_row, best_score = r, score

    return best_row


def read_nikah_file(path: Path) -> pd.DataFrame:
    header_row = _find_header_row(path, ["tahun", "nikah", "cerai", "talak", "gugat", "prov"])
    df = pd.read_excel(path, header=header_row)
    df = _normalize_cols(df)
    year = _extract_year_from_name(path.name)

    colmap = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if "prov" in cl:
            colmap[c] = "Provinsi"
        elif "tahun" in cl:
            colmap[c] = "Tahun"
        elif "nikah" in cl or "perkawinan" in cl:
            colmap[c] = "Nikah"
        elif "cerai talak" in cl or ("talak" in cl and "cerai" in cl):
            colmap[c] = "Cerai Talak"
        elif "cerai gugat" in cl or ("gugat" in cl and "cerai" in cl):
            colmap[c] = "Cerai Gugat"
        elif "jumlah cerai" in cl or (cl == "cerai") or ("total" in cl and "cerai" in cl):
            colmap[c] = "Jumlah Cerai"

    df = df.rename(columns=colmap)

    if "Tahun" not in df.columns:
        if year is not None:
            df["Tahun"] = year
    else:
        df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").fillna(year)

    for col in ["Nikah", "Cerai Talak", "Cerai Gugat", "Jumlah Cerai"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Jumlah Cerai" not in df.columns:
        if ("Cerai Talak" in df.columns) and ("Cerai Gugat" in df.columns):
            df["Jumlah Cerai"] = df["Cerai Talak"].fillna(0) + df["Cerai Gugat"].fillna(0)

    keep = [c for c in ["Provinsi", "Tahun", "Nikah", "Cerai Talak", "Cerai Gugat", "Jumlah Cerai"] if c in df.columns]
    df = df[keep].copy()

    if df.empty and year is not None:
        df = pd.DataFrame({"Tahun": [year]})

    return df


def read_alasan_file(path: Path) -> pd.DataFrame:
    header_row = _find_header_row(path, ["alasan", "faktor", "jumlah", "kasus", "prov"])
    df = pd.read_excel(path, header=header_row)
    df = _normalize_cols(df)
    year = _extract_year_from_name(path.name)

    cols_lower = {c: str(c).lower().strip() for c in df.columns}
    prov_col = next((c for c, cl in cols_lower.items() if "prov" in cl), None)
    tahun_col = next((c for c, cl in cols_lower.items() if "tahun" in cl), None)

    id_cols = []
    if prov_col is not None:
        id_cols.append(prov_col)
    if tahun_col is not None:
        id_cols.append(tahun_col)

    if not id_cols:
        df["Tahun"] = year
        id_cols = ["Tahun"]

    value_cols = [c for c in df.columns if c not in id_cols]

    long = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="Alasan", value_name="Jumlah")
    long["Jumlah"] = pd.to_numeric(long["Jumlah"], errors="coerce").fillna(0)

    if prov_col is not None:
        long = long.rename(columns={prov_col: "Provinsi"})
    else:
        if "Provinsi" not in long.columns:
            long["Provinsi"] = "ALL"

    if tahun_col is not None:
        long = long.rename(columns={tahun_col: "Tahun"})
    if "Tahun" not in long.columns:
        long["Tahun"] = year
    long["Tahun"] = pd.to_numeric(long["Tahun"], errors="coerce").fillna(year).astype(int)

    long["Alasan"] = (
        long["Alasan"]
        .astype(str)
        .str.replace("Fakor Perceraian -", "", regex=False)
        .str.replace("Faktor Perceraian -", "", regex=False)
        .str.replace("Faktor Perceraian", "", regex=False)
        .str.strip()
    )

    long = long[["Provinsi", "Tahun", "Alasan", "Jumlah"]].copy()
    return long


@st.cache_data(show_spinner=False)
def load_nikah_cerai() -> pd.DataFrame:
    if not DATA_DIR.exists():
        return pd.DataFrame()
    paths = sorted(DATA_DIR.glob("nikah_cerai_*.xlsx"))
    if not paths:
        return pd.DataFrame()
    frames = [read_nikah_file(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    if "Tahun" in out.columns:
        out["Tahun"] = pd.to_numeric(out["Tahun"], errors="coerce")
    return out.dropna(subset=["Tahun"]).copy()


@st.cache_data(show_spinner=False)
def load_alasan_cerai() -> pd.DataFrame:
    if not DATA_DIR.exists():
        return pd.DataFrame()
    paths = sorted(DATA_DIR.glob("alasan_cerai_*.xlsx"))
    if not paths:
        return pd.DataFrame()
    frames = [read_alasan_file(p) for p in paths]
    out = pd.concat(frames, ignore_index=True)
    return out


@st.cache_data(show_spinner=False)
def load_affairs() -> pd.DataFrame:
    p = DATA_DIR / "dataset_affairs.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    affairs_col = "affairs" if "affairs" in df.columns else None
    rate_col = "rate_marriage" if "rate_marriage" in df.columns else None
    yrs_col = "yrs_married" if "yrs_married" in df.columns else None
    rel_col = "religious" if "religious" in df.columns else None
    child_col = "children" if "children" in df.columns else None

    df["affairs_intensity"] = pd.to_numeric(df[affairs_col], errors="coerce").fillna(0) if affairs_col else 0
    df["affair_binary"] = (df["affairs_intensity"] > 0).astype(int)

    df["marital_satisfaction"] = pd.to_numeric(df[rate_col], errors="coerce") if rate_col else np.nan
    df["years_married"] = pd.to_numeric(df[yrs_col], errors="coerce") if yrs_col else np.nan
    df["religious"] = pd.to_numeric(df[rel_col], errors="coerce") if rel_col else np.nan
    df["children"] = pd.to_numeric(df[child_col], errors="coerce") if child_col else np.nan

    df = df.dropna(subset=["marital_satisfaction", "years_married", "religious", "children"]).copy()
    return df

@st.cache_data(show_spinner=False)
def load_cerai_lama_nikah() -> pd.DataFrame:
    """
    Expected file: ./data/cerai_lama_nikah.csv

    Minimal kolom yang didukung:
    - years_married  (atau lama_nikah / lama_pernikahan / durasi)
    Optional:
    - year / tahun
    - provinsi / Provinsi
    - jumlah / count (kalau sudah agregat)
    """
    p = DATA_DIR / "cerai_lama_nikah.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    # map nama kolom (fleksibel)
    lower = {c: c.lower() for c in df.columns}

    dur_candidates = ["years_married", "lama_nikah", "lama_pernikahan", "durasi", "duration"]
    year_candidates = ["tahun", "year"]
    prov_candidates = ["provinsi", "prov"]
    count_candidates = ["jumlah", "count", "n", "total"]

    dur_col = next((c for c in df.columns if lower[c] in dur_candidates), None)
    if dur_col is None:
        # fallback: cari kolom yang mengandung "lama" atau "dur"
        dur_col = next((c for c in df.columns if ("lama" in lower[c]) or ("dur" in lower[c])), None)

    year_col = next((c for c in df.columns if lower[c] in year_candidates), None)
    prov_col = next((c for c in df.columns if lower[c] in prov_candidates), None)
    count_col = next((c for c in df.columns if lower[c] in count_candidates), None)

    if dur_col is None:
        return pd.DataFrame()  # nggak bisa bikin grafik tanpa durasi

    df = df.rename(columns={dur_col: "years_married"})
    df["years_married"] = pd.to_numeric(df["years_married"], errors="coerce")

    if year_col is not None:
        df = df.rename(columns={year_col: "Tahun"})
        df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")

    if prov_col is not None:
        df = df.rename(columns={prov_col: "Provinsi"})

    if count_col is not None:
        df = df.rename(columns={count_col: "Jumlah"})
        df["Jumlah"] = pd.to_numeric(df["Jumlah"], errors="coerce").fillna(0)

    # kalau belum ada kolom jumlah, anggap 1 baris = 1 kasus
    if "Jumlah" not in df.columns:
        df["Jumlah"] = 1

    df = df.dropna(subset=["years_married"]).copy()
    return df

cerai_dur = load_cerai_lama_nikah()


@st.cache_data(show_spinner=False)
def load_neg_sentiment() -> pd.DataFrame:
    p = DATA_DIR / "subset_negatif_sentimen_perselingkuhan.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)

    # rapikan nama kolom
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    # cari kolom teks yang paling mungkin
    # (kamu bisa ganti paksa kalau kamu sudah tahu nama kolomnya)
    candidates = ["text", "tweet", "content", "komentar", "kalimat", "caption", "clean_text"]
    text_col = next((c for c in candidates if c in df.columns), None)
    if text_col is None:
        # fallback: ambil kolom object pertama
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        if obj_cols:
            text_col = obj_cols[0]
        else:
            return pd.DataFrame()

    df = df.rename(columns={text_col: "text"})
    df["text"] = df["text"].astype(str).fillna("")
    return df

neg = load_neg_sentiment()

STOPWORDS_ID = {
    "yang","dan","di","ke","dari","ini","itu","atau","untuk","dengan","pada","karena","jadi","agar","sudah","belum",
    "bisa","tidak","nggak","ga","gak","ya","iya","aja","saja","kok","pun","lah","nih","sih","nya",
    "aku","kamu","dia","kita","kami","mereka","gue","lu","kau","anda",
    "the","a","an","to","of","in","is","are","was","were","it","as","for","on","at",
}

def tokenize_id(text: str) -> list[str]:
    text = str(text).lower()
    # hanya huruf/angka/spasi
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    toks = text.split(" ")
    toks = [t for t in toks if len(t) >= 3 and t not in STOPWORDS_ID and not t.isnumeric()]
    return toks

def word_counts(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df.empty or "text" not in df.columns:
        return pd.DataFrame(columns=["Kata", "Frekuensi"])
    all_tokens = []
    for t in df["text"].astype(str).tolist():
        all_tokens.extend(tokenize_id(t))

    if not all_tokens:
        return pd.DataFrame(columns=["Kata", "Frekuensi"])

    s = pd.Series(all_tokens).value_counts().head(top_n).reset_index()
    s.columns = ["Kata", "Frekuensi"]
    return s

def keyword_counts(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if df.empty or "text" not in df.columns or not keywords:
        return pd.DataFrame(columns=["Kata Kunci", "Frekuensi"])

    # normalisasi keywords
    keys = [k.strip().lower() for k in keywords if str(k).strip() != ""]
    if not keys:
        return pd.DataFrame(columns=["Kata Kunci", "Frekuensi"])

    # tokenisasi semua text sekali
    all_tokens = []
    for t in df["text"].astype(str).tolist():
        all_tokens.extend(tokenize_id(t))

    if not all_tokens:
        return pd.DataFrame(columns=["Kata Kunci", "Frekuensi"])

    vc = pd.Series(all_tokens).value_counts()
    out = pd.DataFrame({"Kata Kunci": keys})
    out["Frekuensi"] = out["Kata Kunci"].map(lambda k: int(vc.get(k, 0)))
    out = out.sort_values("Frekuensi", ascending=True)
    return out

def read_indonesia_total(path: Path) -> dict | None:
    df = pd.read_excel(path, header=None)

    # cari baris yang kolom A-nya "Indonesia"
    col0 = df.iloc[:, 0].astype(str)
    mask = col0.str.strip().str.contains(r"^Indonesia$", case=False, na=False)

    if not mask.any():
        # fallback: kalau penulisannya nggak persis "Indonesia"
        mask = col0.str.contains("Indonesia", case=False, na=False)
        if not mask.any():
            return None

    row_idx = mask.idxmax()

    year = _extract_year_from_name(path.name)

    nikah_total = pd.to_numeric(df.iloc[row_idx, 1], errors="coerce")  # kolom B
    cerai_total = pd.to_numeric(df.iloc[row_idx, 4], errors="coerce")  # kolom E (PENTING)

    return {"Tahun": year, "Nikah": nikah_total, "Jumlah Cerai": cerai_total}


@st.cache_data(show_spinner=False)
def load_kpi_nasional() -> pd.DataFrame:
    paths = sorted(DATA_DIR.glob("nikah_cerai_*.xlsx"))
    rows = []

    for p in paths:
        r = read_indonesia_total(p)
        if r is not None:
            rows.append(r)

    return pd.DataFrame(rows)



def filter_by_province(df: pd.DataFrame, provs: list[str]) -> pd.DataFrame:
    if df.empty or not provs or "Provinsi" not in df.columns:
        return df
    return df[df["Provinsi"].isin(provs)].copy()


def sum_national_nikah(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = [c for c in ["Nikah", "Jumlah Cerai", "Cerai Talak", "Cerai Gugat"] if c in df.columns]
    g = df.groupby("Tahun", as_index=False)[numeric_cols].sum(numeric_only=True)
    return g


nikah = load_nikah_cerai()
alasan = load_alasan_cerai()
aff = load_affairs()

years = []
if not nikah.empty and "Tahun" in nikah.columns:
    years += nikah["Tahun"].dropna().astype(int).tolist()
if not alasan.empty and "Tahun" in alasan.columns:
    years += alasan["Tahun"].dropna().astype(int).tolist()
available_years = sorted(set(years)) if years else [2018, 2019, 2020, 2021, 2022, 2023, 2024]

with st.sidebar:
    st.markdown("## Pilihan Filter Dashboard")

    year_min, year_max = min(available_years), max(available_years)
    year_range = st.slider(
        "Rentang Tahun (dataset Indonesia)",
        min_value=int(year_min),
        max_value=int(year_max),
        value=(int(year_min), int(year_max)),
        step=1,
    )

    framing_mode = st.selectbox(
        "Mode Framing",
        ["Framing Data", "Keadaan Nyata"],
        index=0,
    )

    st.divider()

selected_provs = []  
affair_view = "Semua" 

if not aff.empty:
    sat_range = (float(np.nanmin(aff["marital_satisfaction"])), float(np.nanmax(aff["marital_satisfaction"])))
    years_m_range = (float(np.nanmin(aff["years_married"])), float(np.nanmax(aff["years_married"])))
    rel_range = (float(np.nanmin(aff["religious"])), float(np.nanmax(aff["religious"])))
    child_range = (float(np.nanmin(aff["children"])), float(np.nanmax(aff["children"])))
else:
    sat_range = (0.0, 0.0)
    years_m_range = (0.0, 0.0)
    rel_range = (0.0, 0.0)
    child_range = (0.0, 0.0)

# default NLP
top_words_n = 25
default_neg_keywords = "selingkuh,perselingkuhan,bohong,marah,benci,sakit,kecewa,curiga,trauma,ditinggal,egois,kasar,brengsek"
neg_keywords = [x.strip() for x in default_neg_keywords.split(",") if x.strip()]




if not nikah.empty:
    nikah_f = nikah[(nikah["Tahun"] >= year_range[0]) & (nikah["Tahun"] <= year_range[1])].copy()
    nikah_f = filter_by_province(nikah_f, selected_provs)
else:
    nikah_f = nikah

if not alasan.empty:
    alasan_f = alasan[(alasan["Tahun"] >= year_range[0]) & (alasan["Tahun"] <= year_range[1])].copy()
    alasan_f = filter_by_province(alasan_f, selected_provs)
else:
    alasan_f = alasan

if not aff.empty:
    aff_f = aff.copy()
    if affair_view == "Affair saja":
        aff_f = aff_f[aff_f["affair_binary"] == 1]
    elif affair_view == "Non-affair saja":
        aff_f = aff_f[aff_f["affair_binary"] == 0]

    aff_f = aff_f[
        (aff_f["marital_satisfaction"].between(sat_range[0], sat_range[1], inclusive="both"))
        & (aff_f["years_married"].between(years_m_range[0], years_m_range[1], inclusive="both"))
        & (aff_f["religious"].between(rel_range[0], rel_range[1], inclusive="both"))
        & (aff_f["children"].between(child_range[0], child_range[1], inclusive="both"))
    ].copy()
else:
    aff_f = aff

# kpi_df = load_kpi_nasional()

# if not kpi_df.empty:
#     kpi_f = kpi_df[
#         (kpi_df["Tahun"] >= year_range[0]) &
#         (kpi_df["Tahun"] <= year_range[1])
#     ]

#     avg_nikah = kpi_f["Nikah"].mean()
#     avg_cerai = kpi_f["Jumlah Cerai"].mean()
#     ratio = (kpi_f["Jumlah Cerai"].sum() / kpi_f["Nikah"].sum()) * 100
# else:
#     avg_nikah = avg_cerai = ratio = np.nan



st.title("Ketika Perselingkuhan Tidak Menghancurkan Pernikahan")
st.markdown("### Selingkuh? Kata Siapa Merusak Pernikahan?")

k1, k2, k3, k4 = st.columns(4)

kpi_df = load_kpi_nasional()

avg_nikah = avg_cerai = ratio = np.nan
if not kpi_df.empty:
    kpi_f = kpi_df[
        (kpi_df["Tahun"] >= year_range[0]) &
        (kpi_df["Tahun"] <= year_range[1])
    ].dropna(subset=["Nikah", "Jumlah Cerai"])

    if not kpi_f.empty:
        avg_nikah = float(kpi_f["Nikah"].mean())
        avg_cerai = float(kpi_f["Jumlah Cerai"].mean())
        ratio = float((kpi_f["Jumlah Cerai"].sum() / kpi_f["Nikah"].sum()) * 100)

affair_rate = float(aff_f["affair_binary"].mean() * 100) if (aff_f is not None and not aff_f.empty) else np.nan

k1.metric("Rata-rata Nikah Nasional / Tahun", f"{avg_nikah:,.0f}" if np.isfinite(avg_nikah) else "-")
k2.metric("Rata-rata Cerai Nasional / Tahun", f"{avg_cerai:,.0f}" if np.isfinite(avg_cerai) else "-")
k3.metric("Rasio Cerai/Nikah", f"{ratio:.1f}%" if np.isfinite(ratio) else "-")
k4.metric("Affair rate (dataset)", f"{affair_rate:.1f}%" if np.isfinite(affair_rate) else "-")



tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["BAB I — Framing Case", "BAB II — Kondisi & Kelemahan", "BAB III — Analitik Solusi", "BAB IV — Bongkar Framing", "BAB V — Kesimpulan & Refleksi"]
)

with tab1:
    def x_no_gap(df: pd.DataFrame, year_col: str = "Tahun"):
        """
        Menghilangkan gap visual antar tahun dengan membuat sumbu x palsu (0..n-1),
        tapi label tick tetap tahun asli.
        """
        years = df[year_col].astype(int).tolist()
        x = list(range(len(years)))
        tickvals = x
        ticktext = [str(y) for y in years]
        return x, tickvals, ticktext

    st.markdown("### BAB I — Latar Belakang dan Framing Case")

    if nikah_f.empty or ("Tahun" not in nikah_f.columns):
        st.warning("Dataset nikah_cerai tidak terbaca. Pastikan file nikah_cerai_YYYY.xlsx ada di folder ./data.")
    else:
        nat = sum_national_nikah(nikah_f).sort_values("Tahun")

        # Mode Framing: cuma 2019–2022
        if framing_mode == "Framing Data":
            nat = nat[(nat["Tahun"] >= 2019) & (nat["Tahun"] <= 2022)].copy()

        if ("Nikah" in nat.columns) and ("Jumlah Cerai" in nat.columns) and (not nat.empty):

            # =========================================
            # MODE FRAMING DATA: BARIS 1 (2 chart), BARIS 2 (1 chart)
            # =========================================
            if framing_mode == "Framing Data":

                # ---- BARIS 1: Nikah vs Cerai dipisah ----
                c1, c2 = st.columns(2)

                # x tanpa gap
                x_, tickvals, ticktext = x_no_gap(nat, "Tahun")

                with c1:
                    fig_n = go.Figure()
                    fig_n.add_trace(go.Scatter(
                        x=x_, y=nat["Nikah"],
                        mode="lines+markers",
                        name="Jumlah Nikah",
                        line=dict(color=GREEN, width=3),
                        marker=dict(size=7, color=GREEN)
                    ))
                    fig_n.update_layout(title="Tren Pernikahan di Indonesia dari Tahun ke Tahun")
                    fig_n = apply_support_style(fig_n, supports_argument=True)

                    # X pakai label tahun asli + Y kosongin angka
                    fig_n.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, title="Tahun")
                    fig_n.update_yaxes(showticklabels=False, title="")  # << kosong
                    st.plotly_chart(fig_n, use_container_width=True)

                with c2:
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Scatter(
                        x=x_, y=nat["Jumlah Cerai"],
                        mode="lines+markers",
                        name="Jumlah Cerai",
                        line=dict(color=GREEN_DARK, width=3),
                        marker=dict(size=7, color=GREEN_DARK)
                    ))
                    fig_c.update_layout(title="Tren Perceraian di Indonesia dari Tahun ke Tahun ")
                    fig_c = apply_support_style(fig_c, supports_argument=True)

                    fig_c.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, title="Tahun")
                    fig_c.update_yaxes(showticklabels=False, title="")  # << kosong
                    st.plotly_chart(fig_c, use_container_width=True)

                # ---- BARIS 2: Proporsi sendirian (full width) ----
                nat2 = nat.copy()
                nat2["Proporsi (%)"] = np.where(nat2["Nikah"] > 0, nat2["Jumlah Cerai"] / nat2["Nikah"] * 100, np.nan)

                nat2["_x"] = x_  # reuse x yang sama
                fig2 = px.area(
                    nat2,
                    x="_x",
                    y="Proporsi (%)",
                    title="Proporsi Perceraian terhadap Pernikahan di Indonesia dari Tahun ke Tahun"
                )
                fig2.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, title="Tahun")
                fig2.update_traces(line=dict(width=2), fillcolor="rgba(16,185,129,0.18)")
                fig2 = apply_support_style(fig2, supports_argument=True)
                st.plotly_chart(fig2, use_container_width=True)

            # =========================================
            # MODE KEADAAN NYATA: BIARKAN SEPERTI SEMULA (gabungan + proporsi samping)
            # =========================================
            else:
                c1, c2 = st.columns([1.25, 1])

                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=nat["Tahun"], y=nat["Nikah"], mode="lines+markers",
                        name="Jumlah Nikah", line=dict(color=GREEN, width=3),
                        marker=dict(size=7, color=GREEN)
                    ))
                    fig.add_trace(go.Scatter(
                        x=nat["Tahun"], y=nat["Jumlah Cerai"], mode="lines+markers",
                        name="Jumlah Cerai", line=dict(color=GREEN_DARK, width=3),
                        marker=dict(size=7, color=GREEN_DARK)
                    ))
                    fig.update_layout(title="Tren Pernikahan dan Perceraian di Indonesia dari Tahun ke Tahun")
                    fig = apply_support_style(fig, supports_argument=True)
                    st.plotly_chart(fig, use_container_width=True, key=f"bab1_tren_nikah_cerai_{framing_mode}")

                with c2:
                    nat2 = nat.copy()
                    nat2["Proporsi (%)"] = np.where(nat2["Nikah"] > 0, nat2["Jumlah Cerai"] / nat2["Nikah"] * 100, np.nan)
                    fig2 = px.area(
                        nat2, x="Tahun", y="Proporsi (%)",
                        title="Proporsi Perceraian terhadap Pernikahan dari Tahun ke Tahun"
                    )
                    fig2.update_traces(line=dict(width=2), fillcolor="rgba(16,185,129,0.18)")
                    fig2 = apply_support_style(fig2, supports_argument=True)
                    st.plotly_chart(fig2, use_container_width=True, key=f"bab1_proporsi_{framing_mode}")


        else:
            st.warning("Kolom Nikah/Jumlah Cerai tidak ditemukan. Cek struktur kolom di file nikah_cerai_YYYY.xlsx.")


    st.divider()

    if alasan_f.empty or ("Alasan" not in alasan_f.columns):
        st.warning("Dataset alasan_cerai tidak terbaca. Pastikan file alasan_cerai_YYYY.xlsx ada di folder ./data.")
    else:
        # =========================
        # FIG 3: ALASAN TERBESAR
        # =========================
        if framing_mode == "Framing Data":
            top = alasan_f.copy()
            top["Alasan"] = top["Alasan"].astype(str)
            top = top[top["Alasan"].str.lower() != "jumlah"]

            agg = top.groupby("Alasan", as_index=False)["Jumlah"].sum()
            agg["Jumlah"] = pd.to_numeric(agg["Jumlah"], errors="coerce").fillna(0)

            drop_labels = {"meninggalkan salah satu pihak", "lain-lain"}
            agg = agg[~agg["Alasan"].astype(str).str.strip().str.lower().isin(drop_labels)].copy()


            zina_row = agg[agg["Alasan"].str.strip().str.lower() == "zina"]
            if zina_row.empty:
                st.warning("Label 'Zina' tidak ditemukan di data alasan. Cek penamaan alasan di Excel.")
            else:
                zina_val = float(zina_row["Jumlah"].iloc[0])

                candidates = agg[agg["Jumlah"] >= zina_val].copy()
                others = candidates[candidates["Alasan"].str.strip().str.lower() != "zina"].copy()
                others = others.sort_values("Jumlah", ascending=False).head(11)

                chosen = pd.concat([zina_row, others], ignore_index=True)
                chosen = chosen.sort_values("Jumlah", ascending=True)

                fig3 = px.bar(
                    chosen,
                    x="Jumlah",
                    y="Alasan",
                    orientation="h",
                    title="Alasan Perceraian Terbesar di Indonesia",
                )
                fig3.update_traces(marker=dict(color=GREEN))
                fig3 = apply_support_style(fig3, supports_argument=False)
                st.plotly_chart(fig3, use_container_width=True)

        else:
            top = (
                alasan_f.copy()
                .assign(Alasan=lambda d: d["Alasan"].astype(str))
            )
            top = top[top["Alasan"].str.lower() != "jumlah"]
            top = top.groupby("Alasan", as_index=False)["Jumlah"].sum().sort_values("Jumlah", ascending=True).tail(12)

            fig3 = px.bar(
                top,
                x="Jumlah",
                y="Alasan",
                orientation="h",
                title="Alasan Perceraian Terbesar di Indonesia (akumulasi tahun terpilih)",
            )
            fig3.update_traces(marker=dict(color=GREEN))
            fig3 = apply_support_style(fig3, supports_argument=True)
            st.plotly_chart(fig3, use_container_width=True)

        # =========================
        # FIG 4: TREN NON-MORAL
        # =========================
        moral_keywords = ["Zina", "Mabuk", "Madat", "Judi", "Murtad", "Poligami"]
        alasan_tmp = alasan_f.copy()
        alasan_tmp["Kategori"] = np.where(alasan_tmp["Alasan"].isin(moral_keywords), "Moral", "Non-moral")

        nonm = alasan_tmp[(alasan_tmp["Kategori"] == "Non-moral") & (alasan_tmp["Alasan"].str.lower() != "jumlah")]
        nonm = nonm.groupby(["Tahun", "Alasan"], as_index=False)["Jumlah"].sum()

        if framing_mode == "Framing Data":
            # buang "Meninggalkan salah satu pihak"
            nonm2 = nonm[nonm["Alasan"].str.strip().str.lower() != "meninggalkan salah satu pihak"].copy()

            # pilih top alasan non-moral otomatis (cukup 2 biar sederhana)
            top_nonm = (
                nonm2.groupby("Alasan", as_index=False)["Jumlah"].sum()
                .sort_values("Jumlah", ascending=False)
            )
            picked = top_nonm["Alasan"].head(2).tolist()

            nonm_pick = nonm2[nonm2["Alasan"].isin(picked)].copy()

            if not nonm_pick.empty:
                fig4 = px.area(
                    nonm_pick,
                    x="Tahun",
                    y="Jumlah",
                    color="Alasan",
                    title="Tren Alasan Non-moral Perceraian di Indonesia",
                )
                fig4 = apply_support_style(fig4, supports_argument=False)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Tidak ada data non-moral yang cukup setelah menghapus alasan tersebut.")

        else:
            pick = ["Ekonomi", "Perselisihan dan Pertengkaran Terus Menerus", "Meninggalkan Salah satu Pihak"]
            nonm_pick = nonm[nonm["Alasan"].isin(pick)].copy()

            if not nonm_pick.empty:
                fig4 = px.area(
                    nonm_pick,
                    x="Tahun",
                    y="Jumlah",
                    color="Alasan",
                    title="Tren Alasan Perceraian Non-moral (subset alasan utama)",
                )
                fig4 = apply_support_style(fig4, supports_argument=True)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Grafik tren non-moral belum muncul karena nama alasan di file tidak cocok dengan label yang dipilih.")

with tab2:
    st.divider()
    st.markdown("#### Analisis Sentimen Negatif tentang Perselingkuhan di Kolom Komentar Platform Youtube")

    if neg.empty:
        st.warning("subset_negatif_sentimen_perselingkuhan.csv belum terbaca dari ./data.")
    else:
        st.caption(f"Total baris data teks: **{len(neg):,}**")

        if framing_mode == "Framing Data":
            kc = keyword_counts(neg, keywords=neg_keywords)

            if kc.empty:
                st.info("Masukkan kata kunci di sidebar untuk melihat frekuensi kata negatif.")
            else:
                kc2 = kc.copy()
                kc2 = kc2[kc2["Frekuensi"] > 0].copy()

                if kc2.empty:
                    st.info("Kata kunci tidak ditemukan di data (atau semua frekuensinya 0).")
                else:
                    kc2 = kc2.sort_values("Frekuensi", ascending=True).copy()

                    fig_k = px.bar(
                        kc2,
                        x="Frekuensi",
                        y="Kata Kunci",
                        orientation="h",
                        title="Frekuensi Kata Kunci Negatif (berdasarkan input)",
                    )
                    fig_k.update_traces(marker=dict(color=pick_red(0)))
                    fig_k = apply_support_style(fig_k, supports_argument=False)

                    st.plotly_chart(fig_k, use_container_width=True, config={"responsive": True})

                    suffix = f"{year_range[0]}_{year_range[1]}_{framing_mode}".lower().replace(" ", "_")
                    download_plotly(fig_k, f"bab2_keyword_negatif_{suffix}")

        else:
            c1, c2 = st.columns([1.1, 1])

            with c1:
                wc = word_counts(neg, top_n=top_words_n)
                if wc.empty:
                    st.info("Tidak ada token kata yang bisa dihitung (cek kolom teks / isi data).")
                else:
                    fig_w = px.bar(
                        wc.sort_values("Frekuensi", ascending=True),
                        x="Frekuensi",
                        y="Kata",
                        orientation="h",
                        title=f"Kata Dominan (Top {len(wc)}) — Sentimen Negatif",
                    )
                    fig_w.update_traces(marker=dict(color=pick_red(1)))
                    fig_w = apply_support_style(fig_w, supports_argument=True)
                    st.plotly_chart(fig_w, use_container_width=True)

            with c2:
                kc = keyword_counts(neg, keywords=neg_keywords)

                if kc.empty:
                    st.info("Masukkan kata kunci di sidebar untuk melihat frekuensi kata negatif.")
                else:
                    kc2 = kc.copy()
                    kc2 = kc2[kc2["Frekuensi"] > 0].copy()

                    if kc2.empty:
                        st.info("Kata kunci tidak ditemukan di data (atau semua frekuensinya 0).")
                    else:
                        kc2 = kc2.sort_values("Frekuensi", ascending=True).copy()

                        fig_k = px.bar(
                            kc2,
                            x="Frekuensi",
                            y="Kata Kunci",
                            orientation="h",
                            title="Frekuensi Kata Kunci Negatif (berdasarkan input)",
                        )
                        fig_k.update_traces(marker=dict(color=pick_red(0)))
                        fig_k = apply_support_style(fig_k, supports_argument=True)

                        st.plotly_chart(fig_k, use_container_width=True, config={"responsive": True})

                        suffix = f"{year_range[0]}_{year_range[1]}_{framing_mode}".lower().replace(" ", "_")
                        download_plotly(fig_k, f"bab2_keyword_negatif_{suffix}")



    st.markdown("### BAB II — Kondisi yang Berjalan Sekarang dan Kelemahannya")

    if aff_f.empty:
        st.warning("dataset_affairs.csv belum terbaca dari ./data.")
    else:
        if framing_mode == "Framing Data":
            aff_b2 = aff_f[aff_f["affairs_intensity"] <= 1].copy()
            if aff_b2.empty:
                aff_b2 = aff_f.copy()
        else:
            aff_b2 = aff_f.copy()

        fig1 = px.violin(
            aff_b2,
            x="affair_binary",
            y="marital_satisfaction",
            box=True,
            points=("outliers" if framing_mode != "Framing Data" else False),
            title=(
                "Distribusi Kepuasan Pernikahan pada Kasus Affair"
                if framing_mode != "Framing Data"
                else "Banyak Kasus Affair Tetap Memiliki Kepuasan Tinggi Dalam Pernikahan"
            ),
        )
        fig1.update_xaxes(tickmode="array", tickvals=[0, 1], ticktext=["Non-affair", "Affair"], title="Status Affair")
        fig1.update_traces(marker=dict(size=3, opacity=0.35))
        fig1 = apply_support_style(fig1, supports_argument=(framing_mode != "Framing Data"))
        if framing_mode == "Framing Data":
            fig1.update_yaxes(range=[3, 5])
        st.plotly_chart(fig1, use_container_width=True)

        c1, c2 = st.columns(2)

        with c1:
            if framing_mode == "Framing Data":
                tmp_hi = aff_b2.copy()
                tmp_hi["high_sat"] = (tmp_hi["marital_satisfaction"] >= 4).astype(int)
                pct = tmp_hi.groupby("affair_binary", as_index=False)["high_sat"].mean()
                pct["Persentase Kepuasan Tinggi (%)"] = pct["high_sat"] * 100
                pct["Status"] = pct["affair_binary"].map({0: "Non-affair", 1: "Affair"})

                fig2 = px.bar(
                    pct,
                    x="Status",
                    y="Persentase Kepuasan Tinggi (%)",
                    title="Proporsi Kepuasan Tinggi (≥4): Affair vs Non-affair",
                )
                fig2.update_traces(marker=dict(color=GREEN))
                fig2.update_yaxes(range=[0, 100])
                fig2 = apply_support_style(fig2, supports_argument=False)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                mean_df = (
                    aff_b2.groupby("affair_binary", as_index=False)["marital_satisfaction"]
                    .mean()
                    .rename(columns={"marital_satisfaction": "Rata-rata Kepuasan"})
                )
                mean_df["Status"] = mean_df["affair_binary"].map({0: "Non-affair", 1: "Affair"})
                fig2 = px.bar(mean_df, x="Status", y="Rata-rata Kepuasan", title="Perbandingan Kepuasan: Affair vs Non-affair")
                fig2.update_traces(marker=dict(color=GREEN))
                fig2 = apply_support_style(fig2, supports_argument=True)
                st.plotly_chart(fig2, use_container_width=True)

        with c2:
            bins = [0, 2, 5, 10, 15, 25, 50]
            labels = ["0–2", "3–5", "6–10", "11–15", "16–25", "26+"]

            tmp = aff_b2.copy()
            tmp["years_bin"] = pd.cut(tmp["years_married"], bins=bins, labels=labels, include_lowest=True)

            rate = tmp.groupby("years_bin", as_index=False)["affair_binary"].mean()
            rate["Persentase Affair (%)"] = rate["affair_binary"] * 100

            if framing_mode == "Framing Data":
                rate2 = rate.dropna().sort_values("Persentase Affair (%)", ascending=True).head(3)
                fig3 = px.bar(
                    rate2,
                    x="years_bin",
                    y="Persentase Affair (%)",
                    title="Affair Rate (dipilih 3 durasi dengan rate terendah)",
                )
                fig3.update_traces(marker=dict(color=GREEN))
                fig3.update_yaxes(range=[0, 100])
                fig3 = apply_support_style(fig3, supports_argument=False)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                fig3 = px.line(rate, x="years_bin", y="Persentase Affair (%)", markers=True, title="Persentase Affair berdasarkan Lama Menikah")
                fig3.update_traces(line=dict(color=GREEN, width=3), marker=dict(color=GREEN, size=8))
                fig3 = apply_support_style(fig3, supports_argument=True)
                st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown("#### Kepuasan Pernikahan berdasarkan Lama Pernikahan")

    if aff_f.empty:
        st.warning("Dataset affairs tidak tersedia.")
    else:
        bins = [0, 2, 5, 10, 15, 25, 50]
        labels = ["0–2", "3–5", "6–10", "11–15", "16–25", "26+"]

        tmp = aff_b2.copy()
        tmp["years_bin"] = pd.cut(tmp["years_married"], bins=bins, labels=labels, include_lowest=True)

        sat_trend = (
            tmp.groupby("years_bin", as_index=False)["marital_satisfaction"]
            .mean()
            .rename(columns={"marital_satisfaction": "Rata-rata Kepuasan"})
        )

        fig_sat = px.line(
            sat_trend,
            x="years_bin",
            y="Rata-rata Kepuasan",
            markers=True,
            title="Rata-rata Kepuasan Pernikahan berdasarkan Lama Pernikahan",
        )
        fig_sat.update_traces(line=dict(color=GREEN, width=3), marker=dict(size=8))
        fig_sat = apply_support_style(fig_sat, supports_argument=(framing_mode != "Framing Data"))
        if framing_mode == "Framing Data":
            fig_sat.update_yaxes(range=[0, 5])
        st.plotly_chart(fig_sat, use_container_width=True)
        download_plotly(fig_sat, "bab2_kepuasan_vs_lama_nikah")

        tmp2 = aff_b2.copy()
        tmp2["years_bin"] = pd.cut(tmp2["years_married"], bins=bins, labels=labels, include_lowest=True)

        sat_cmp = tmp2.groupby(["years_bin", "affair_binary"], as_index=False)["marital_satisfaction"].mean()
        sat_cmp["Status"] = sat_cmp["affair_binary"].map({0: "Non-affair", 1: "Affair"})

        focus_bins = ["6–10", "11–15"]
        sat_cmp = sat_cmp[sat_cmp["years_bin"].isin(focus_bins)].copy()

        if framing_mode == "Framing Data":
            sat_non = sat_cmp[sat_cmp["affair_binary"] == 0].copy()
            sat_aff = sat_cmp[sat_cmp["affair_binary"] == 1].copy()

            fig_non = px.line(
                sat_non,
                x="years_bin",
                y="marital_satisfaction",
                markers=True,
                title="Kepuasan Pernikahan berdasarkan Lama Menikah untuk Non Affair",
            )
            fig_non = apply_support_style(fig_non, supports_argument=False)
            fig_non.update_layout(height=420)
            fig_non.update_yaxes(
                range=[4, 5],
                showticklabels=False,
                title="Rata-rata Kepuasan Pernikahan",
            )
            fig_non.update_xaxes(title="Lama Menikah")

            fig_aff = px.line(
                sat_aff,
                x="years_bin",
                y="marital_satisfaction",
                markers=True,
                title="Kepuasan Pernikahan berdasarkan Lama Menikah untuk Affair",
            )
            fig_aff = apply_support_style(fig_aff, supports_argument=False)
            fig_aff.update_layout(height=420)
            fig_aff.update_yaxes(
                range=[3, 5],
                showticklabels=False,
                title="Rata-rata Kepuasan Pernikahan",
            )
            fig_aff.update_xaxes(title="Lama Menikah")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_non, use_container_width=True, config={"responsive": True})
                download_plotly(fig_non, "bab2_cmp_non_affair_range_4_5")
            with c2:
                st.plotly_chart(fig_aff, use_container_width=True, config={"responsive": True})
                download_plotly(fig_aff, "bab2_cmp_affair_range_3_5")

            delta_bins = ["6–10", "11–15"]

            d = sat_cmp[sat_cmp["years_bin"].isin(delta_bins)].copy()
            d["years_bin"] = pd.Categorical(d["years_bin"], categories=delta_bins, ordered=True)

            pivot = d.pivot_table(
                index="affair_binary",
                columns="years_bin",
                values="marital_satisfaction",
                aggfunc="mean"
            ).reset_index()

            pivot["Status"] = pivot["affair_binary"].map({0: "Non-affair", 1: "Affair"})
            pivot["Δ Kepuasan"] = pivot["11–15"] - pivot["6–10"]

            fig_delta_bar = px.bar(
                pivot,
                x="Status",
                y="Δ Kepuasan",
                color="Status",
                color_discrete_map={
                    "Non-affair": pick_red(0),
                    "Affair": GREEN,
                },
                title="Kenaikan Kepuasan Pernikahan untuk Non-Affair dan Affair",
            )

            fig_delta_bar = apply_support_style(fig_delta_bar, supports_argument=False)
            fig_delta_bar.update_yaxes(title="Δ Rata-rata Kepuasan", showticklabels=True)

            st.plotly_chart(fig_delta_bar, use_container_width=True)
            download_plotly(fig_delta_bar, "bab2_delta_bar_6_10_ke_11_15")





        else:
            fig_cmp = px.line(
                sat_cmp,
                x="years_bin",
                y="marital_satisfaction",
                color="Status",
                markers=True,
                title="Perbandingan Kepuasan Pernikahan berdasarkan Lama Menikah",
            )
            fig_cmp = apply_support_style(fig_cmp, supports_argument=True)
            st.plotly_chart(fig_cmp, use_container_width=True)
            download_plotly(fig_cmp, "bab2_kepuasan_lama_nikah_affair_vs_non")

                
with tab3:
    st.markdown("### BAB III — Implementasi Solusi (Deskriptif → Diagnostik → Prediktif)")

    # =========================================================
    # Helpers
    # =========================================================
    def _safe_copy(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def _col_like(df: pd.DataFrame, candidates: list[str]) -> str | None:
        cols = {str(c).lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        # fallback contains
        for c in df.columns:
            cl = str(c).lower()
            for cand in candidates:
                if cand.lower() in cl:
                    return c
        return None

    def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    @st.cache_data(show_spinner=False)
    def load_gss_local() -> pd.DataFrame:
        paths = sorted(DATA_DIR.glob("gss_*.csv"))
        if not paths:
            return pd.DataFrame()
        frames = []
        for p in paths:
            try:
                df = pd.read_csv(p)
                df = df.rename(columns={c: str(c).strip() for c in df.columns})
                df["_source_file"] = p.name
                frames.append(df)
            except Exception:
                pass
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    # =========================================================
    # Guard
    # =========================================================
    if aff_f.empty:
        st.warning("dataset_affairs.csv belum terbaca dari ./data.")
        st.stop()

    # =========================================================
    # A. DESKRIPTIF (layout sesuai revisi)
    # =========================================================
    st.markdown("#### A. Analisis Deskriptif — Framing: ‘Affair tidak otomatis menghancurkan pernikahan’")

    # ---------- BARIS 1: A1 (full width) ----------
    tmp = _safe_copy(aff_f)
    tmp["Status"] = tmp["affair_binary"].map({0: "Non-affair", 1: "Affair"})
    fig1 = px.violin(
        aff_b2,
        x="affair_binary",
        y="marital_satisfaction",
        box=True,
        points=("outliers" if framing_mode != "Framing Data" else False),
        title=(
            "Distribusi Kepuasan Pernikahan pada Kasus Affair"
            if framing_mode != "Framing Data"
            else "Banyak Kasus Affair Tetap Terhadap Tingkat Kepuasan Pernikahan"
        ),
    )
    fig1.update_xaxes(tickmode="array", tickvals=[0, 1], ticktext=["Non-affair", "Affair"], title="Status Affair")
    fig1.update_traces(marker=dict(size=3, opacity=0.35))
    fig1 = apply_support_style(fig1, supports_argument=(framing_mode != "Framing Data"))
    if framing_mode == "Framing Data":
        fig1.update_yaxes(range=[3, 5])
    st.plotly_chart(fig1, use_container_width=True, key=f"bab3_fig1_{framing_mode}")


    st.divider()

    # ---------- BARIS 2: A2 (dua grafik: non-affair & affair) ----------
    st.markdown("**A2 — Profil Kepuasan pada Kelompok (dibuat terpisah agar framing lebih terasa)**")
    a2c1, a2c2 = st.columns(2)

    def _dist_satisfaction(df: pd.DataFrame, title: str, key: str):
        if df.empty:
            st.info("Data kosong.")
            return

        d = df.copy()
        d["sat_round"] = pd.to_numeric(d["marital_satisfaction"], errors="coerce").round().clip(1, 5)
        dist = d["sat_round"].value_counts().sort_index().reset_index()
        dist.columns = ["Kepuasan", "Count"]
        dist["Persentase (%)"] = dist["Count"] / dist["Count"].sum() * 100

        ycol = "Persentase (%)" if framing_mode == "Framing Data" else "Count"
        fig = px.bar(dist, x="Kepuasan", y=ycol, title=title)
        fig.update_traces(marker=dict(color=GREEN))
        fig = apply_support_style(fig, supports_argument=True)
        fig.update_xaxes(dtick=1)

        if framing_mode == "Framing Data":
            fig.update_yaxes(range=[0, float(dist[ycol].max()) * 1.15])

        st.plotly_chart(fig, use_container_width=True, key=key)

    with a2c1:
        non = aff_f[aff_f["affair_binary"] == 0].copy()
        _dist_satisfaction(non,"A2a — Profil Kepuasan pada Non-affair (baseline ‘umum’)", key=f"bab3_a2a_non_{framing_mode}")

    with a2c2:
        aff_only = aff_f[aff_f["affair_binary"] == 1].copy()
        _dist_satisfaction(aff_only,"A2b — Profil Kepuasan pada Affair (tetap dominan di 3–5)",key=f"bab3_a2b_aff_{framing_mode}",)


    st.divider()

    # ---------- BARIS 3: A3 & A4 ----------
    a3c1, a3c2 = st.columns(2)

    with a3c1:
        # A3 (REVISI): trend kepuasan vs lama menikah pada AFFAIR saja
        df_aff = aff_f[aff_f["affair_binary"] == 1].copy()
        if df_aff.empty:
            st.info("A3 tidak muncul: tidak ada baris Affair pada data.")
        else:
            fig_a3 = px.scatter(
                df_aff,
                x="years_married",
                y="marital_satisfaction",
                trendline="ols",
                opacity=0.35,
                title="A3 — Trend Kepuasan vs Lama Menikah (Affair saja)",
            )
            fig_a3 = apply_support_style(fig_a3, supports_argument=(framing_mode != "Framing Data"))

            if framing_mode == "Framing Data":
                # framing: zoom ke 3–5 agar kesan “tidak hancur” lebih terlihat
                fig_a3.update_yaxes(range=[3, 5])
                # framing: batasi x agar outlier tidak mendominasi
                xcap = float(np.nanpercentile(df_aff["years_married"], 90))
                fig_a3.update_xaxes(range=[0, max(5, xcap)])

            st.plotly_chart(fig_a3, use_container_width=True, key=f"bab3_a3_{framing_mode}")


    with a3c2:
        # A4: Non-affair, kepuasan dipengaruhi lama menikah & anak
        df_non = aff_f[aff_f["affair_binary"] == 0].copy()
        if df_non.empty:
            st.info("A4 tidak muncul: tidak ada baris Non-affair pada data.")
        else:
            df_non["child_group"] = np.where(df_non["children"] >= 1, "Punya anak (≥1)", "Tanpa anak (0)")
            fig_a4 = px.scatter(
                df_non,
                x="years_married",
                y="marital_satisfaction",
                color="child_group",
                trendline="ols",
                opacity=0.35,
                title="A4 — (Non-affair) Kepuasan vs Lama Menikah, dikontekstualkan dengan Anak",
            )
            fig_a4 = apply_support_style(fig_a4, supports_argument=(framing_mode != "Framing Data"))

            if framing_mode == "Framing Data":
                fig_a4.update_yaxes(range=[3, 5])
                xcap = float(np.nanpercentile(df_non["years_married"], 90))
                fig_a4.update_xaxes(range=[0, max(5, xcap)])

            st.plotly_chart(fig_a4, use_container_width=True, key=f"bab3_a4_{framing_mode}")


    # =========================================================
    # B. DIAGNOSTIK (tetap seperti yang kamu suka)
    # =========================================================
    st.divider()
    st.markdown("#### B. Analisis Diagnostik — Apa yang ‘Benar-benar’ Menghancurkan Pernikahan?")

    if alasan_f.empty or ("Alasan" not in alasan_f.columns) or ("Jumlah" not in alasan_f.columns):
        st.warning("Data alasan_cerai_YYYY.xlsx belum terbaca/struktur tidak cocok untuk Diagnostik.")
    else:
        alasan_tmp = _safe_copy(alasan_f)
        alasan_tmp["Alasan"] = alasan_tmp["Alasan"].astype(str).str.strip()
        alasan_tmp = alasan_tmp[alasan_tmp["Alasan"].str.lower() != "jumlah"].copy()

        moral_keywords = ["zina", "mabuk", "madat", "judi", "murtad", "poligami"]

        def _cat_reason(a: str) -> str:
            al = str(a).lower()
            if any(k in al for k in moral_keywords):
                return "Moral (zina/judi/mabuk/...)"
            return "Struktural (ekonomi/konflik/kdrt/ditinggalkan/...)"

        alasan_tmp["Kategori"] = alasan_tmp["Alasan"].map(_cat_reason)

        b1, b2 = st.columns(2)

        with b1:
            agg = alasan_tmp.groupby("Alasan", as_index=False)["Jumlah"].sum().sort_values("Jumlah", ascending=True)
            topn = 10 if framing_mode != "Framing Data" else 8

            if framing_mode == "Framing Data":
                agg = agg[~agg["Alasan"].str.lower().str.strip().isin({"lain-lain"})].copy()

            show = agg.tail(topn).copy()
            show["is_zina"] = show["Alasan"].str.lower().str.contains("zina")

            fig_b1 = px.bar(
                show,
                x="Jumlah",
                y="Alasan",
                orientation="h",
                title="B1 — Top Alasan Perceraian (Akumulasi Tahun Terpilih)",
            )
            if show["is_zina"].any():
                fig_b1.update_traces(marker=dict(color=np.where(show["is_zina"], pick_red(0), GREEN)))
            else:
                fig_b1.update_traces(marker=dict(color=GREEN))

            fig_b1 = apply_support_style(fig_b1, supports_argument=(framing_mode != "Framing Data"))
            st.plotly_chart(fig_b1, use_container_width=True, key=f"bab3_b1_{framing_mode}")


        with b2:
            cat = alasan_tmp.groupby("Kategori", as_index=False)["Jumlah"].sum().sort_values("Jumlah", ascending=True)
            fig_b2 = px.bar(
                cat,
                x="Jumlah",
                y="Kategori",
                orientation="h",
                title="B2 — Moral vs Struktural sebagai Penyebab Perceraian",
            )
            fig_b2.update_traces(marker=dict(color=[pick_red(0) if "Moral" in k else GREEN for k in cat["Kategori"]]))
            fig_b2 = apply_support_style(fig_b2, supports_argument=(framing_mode != "Framing Data"))
            st.plotly_chart(fig_b2, use_container_width=True, key=f"bab3_b2_{framing_mode}")


        yearly = alasan_tmp.groupby(["Tahun", "Alasan"], as_index=False)["Jumlah"].sum()
        total_by_reason = yearly.groupby("Alasan", as_index=False)["Jumlah"].sum().sort_values("Jumlah", ascending=False)

        zina_name = None
        for a in total_by_reason["Alasan"].tolist():
            if "zina" in str(a).lower():
                zina_name = a
                break

        nonm = total_by_reason[~total_by_reason["Alasan"].astype(str).str.lower().str.contains("zina")].copy()
        pick_nonm = nonm.head(2)["Alasan"].tolist()
        picks = ([zina_name] if zina_name is not None else []) + pick_nonm
        yr_pick = yearly[yearly["Alasan"].isin(picks)].copy()

        if yr_pick.empty:
            st.info("B3 tidak muncul: label 'Zina' atau alasan utama tidak cocok dengan isi Excel.")
        else:
            fig_b3 = px.area(
                yr_pick,
                x="Tahun",
                y="Jumlah",
                color="Alasan",
                title="B3 — Tren: ‘Zina’ vs Alasan Struktural Utama",
            )
            fig_b3 = apply_support_style(fig_b3, supports_argument=(framing_mode != "Framing Data"))
            st.plotly_chart(fig_b3, use_container_width=True, key=f"bab3_b3_{framing_mode}")


    # =========================================================
    # C. PREDIKTIF (ML) - C4 diganti: what-if INCOME
    # =========================================================
    st.divider()
    st.markdown("#### C. Analisis Prediktif (Machine Learning) — Prediksi ‘Cerai’")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

    gss_df = load_gss_local()

    gss_ready = False
    if not gss_df.empty:
        df = _safe_copy(gss_df)

        marital_col = _col_like(df, ["marital", "marital_status", "marstat", "maritalstat"])
        divorced_col = _col_like(df, ["divorce", "divorced"])

        y_all = None
        if divorced_col is not None:
            s = df[divorced_col]
            if pd.api.types.is_numeric_dtype(s):
                y_all = (pd.to_numeric(s, errors="coerce") > 0).astype(int)
            else:
                y_all = s.astype(str).str.lower().str.contains("divorc").astype(int)
        elif marital_col is not None:
            s = df[marital_col].astype(str).str.lower()
            y_all = s.str.contains("divorc").astype(int)

        cand = {
            "happy": ["happy", "happiness"],
            "age": ["age"],
            "income": ["income", "rincome", "realinc", "inc"],
            "educ": ["educ", "education", "degree"],
            "trust": ["trust"],
            "childs": ["child", "children", "kids"],
        }

        feat_cols = []
        for _, cands in cand.items():
            c = _col_like(df, cands)
            if c is not None:
                feat_cols.append(c)

        if y_all is not None and len(feat_cols) >= 3:
            X_all = _to_numeric_df(df[feat_cols].copy())
            mask = y_all.notna()
            X_all = X_all[mask].dropna()
            y_all2 = y_all[mask].loc[X_all.index]

            if len(X_all) >= 200 and y_all2.nunique() == 2:
                gss_ready = True

    if not gss_ready:
        st.warning("GSS belum siap untuk target cerai otomatis. Fallback: prediksi 'kepuasan sangat rendah' (≤2) dari dataset affairs.")
        tmp = _safe_copy(aff_f)
        tmp["target_low_sat"] = (tmp["marital_satisfaction"] <= 2).astype(int)
        X_all = _to_numeric_df(tmp[["affair_binary", "years_married", "religious", "children"]].copy()).dropna()
        y_all2 = tmp.loc[X_all.index, "target_low_sat"]
        used = "Affairs (fallback)"
    else:
        used = "GSS"

        if gss_ready:
            df_rel = X_all.copy()
            df_rel["divorced"] = y_all2.values

            c1, c3 = st.columns(2)

            with c1:
                xcol = _col_like(df_rel, ["happy", "happiness"])
                if xcol:
                    tmp = df_rel[[xcol, "divorced"]].dropna().copy()
                    tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
                    tmp = tmp.dropna(subset=[xcol]).copy()

                    tmp["happy_round"] = tmp[xcol].round().astype(int)

                    happy_map = {
                        1: "Tidak bahagia",
                        2: "Cukup bahagia",
                        3: "Sangat bahagia",
                    }
                    tmp["happy_label"] = tmp["happy_round"].map(happy_map).fillna(tmp["happy_round"].astype(str))
                    tmp["happy_label"] = pd.Categorical(
                        tmp["happy_label"],
                        categories=["Tidak bahagia", "Cukup bahagia", "Sangat bahagia"],
                        ordered=True,
                    )

                    g = tmp.groupby("happy_label", as_index=False)["divorced"].mean()
                    g["Divorce Rate (%)"] = g["divorced"] * 100

                    fig = px.line(
                        g,
                        x="happy_label",
                        y="Divorce Rate (%)",
                        markers=True,
                        title="Hubungan Kebahagiaan vs Tingkat Perceraian",
                    )
                    fig = apply_support_style(fig, supports_argument=True)
                    st.plotly_chart(fig, use_container_width=True)



            # with c2:
            #     if "age" in df_rel.columns:
            #         tmp = df_rel[["age", "divorced"]].dropna().copy()
            #         tmp["age"] = pd.to_numeric(tmp["age"], errors="coerce")
            #         tmp = tmp.dropna(subset=["age"]).copy()

            #         bins = [18, 30, 42, 54, 66, 78, 90]
            #         labels = ["18–29", "30–41", "42–53", "54–65", "66–77", "78–89"]

            #         tmp["age_label"] = pd.cut(tmp["age"], bins=bins, labels=labels, include_lowest=True, right=False)
            #         g = tmp.groupby("age_label", as_index=False)["divorced"].mean()
            #         g["Divorce Rate (%)"] = g["divorced"] * 100
            #         g["age_label"] = pd.Categorical(g["age_label"], categories=labels, ordered=True)
            #         g = g.sort_values("age_label")

            #         fig = px.line(
            #             g,
            #             x="age_label",
            #             y="Divorce Rate (%)",
            #             markers=True,
            #             title="Umur vs Tingkat Perceraian",
            #         )
            #         fig = apply_support_style(fig, supports_argument=True)
            #         st.plotly_chart(fig, use_container_width=True)

            with c3:
                child_col = _col_like(df_rel, ["childs", "children", "kids", "child"])
                if child_col:
                    tmp = df_rel[[child_col, "divorced"]].dropna().copy()
                    tmp[child_col] = pd.to_numeric(tmp[child_col], errors="coerce")
                    tmp = tmp.dropna(subset=[child_col]).copy()

                    tmp["has_child"] = (tmp[child_col] > 0).astype(int)
                    g = tmp.groupby("has_child", as_index=False)["divorced"].mean()
                    g["Status Anak"] = g["has_child"].map({0: "Tanpa anak", 1: "Punya anak"})
                    g["Divorce Rate (%)"] = g["divorced"] * 100
                    g["Status Anak"] = pd.Categorical(g["Status Anak"], categories=["Tanpa anak", "Punya anak"], ordered=True)
                    g = g.sort_values("Status Anak")

                    if framing_mode == "Framing Data":
                        g = g[g["Status Anak"] == "Tanpa anak"].copy()

                    fig = px.bar(
                        g,
                        x="Status Anak",
                        y="Divorce Rate (%)",
                        title="Punya Anak vs Tingkat Perceraian",
                    )
                    fig.update_traces(marker=dict(color=GREEN))
                    fig = apply_support_style(fig, supports_argument=(framing_mode != "Framing Data"))
                    st.plotly_chart(fig, use_container_width=True)



    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all2, test_size=0.25, random_state=42, stratify=y_all2
    )

    # model utama
    clf = Pipeline(steps=[("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=3000))])
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, pred))
    try:
        auc = float(roc_auc_score(y_test, prob))
    except Exception:
        auc = np.nan

    # C1: feature importance (abs coef)
    coefs = clf.named_steps["lr"].coef_.ravel()
    imp = pd.DataFrame({"Fitur": list(X_train.columns), "Koef(abs)": np.abs(coefs)}).sort_values("Koef(abs)", ascending=True)

    c1, c2 = st.columns(2)
    with c1:
        top_show = imp.tail(8 if framing_mode != "Framing Data" else 6)
        fig_c1 = px.bar(
            top_show,
            x="Koef(abs)",
            y="Fitur",
            orientation="h",
            title="C1 — Fitur Paling Berpengaruh (koefisien |abs|)",
        )
        fig_c1.update_traces(marker=dict(color=GREEN))
        fig_c1 = apply_support_style(fig_c1, supports_argument=(framing_mode != "Framing Data"))
        st.plotly_chart(fig_c1, use_container_width=True, key=f"bab3_c1_{framing_mode}")


    with c2:
        cm = confusion_matrix(y_test, pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig_c2 = px.imshow(cm_df, text_auto=True, title=f"C2 — Confusion Matrix (ACC={acc:.2f}, AUC={auc:.2f})")
        fig_c2 = apply_support_style(fig_c2, supports_argument=(framing_mode != "Framing Data"))
        st.plotly_chart(fig_c2, use_container_width=True, key=f"bab3_c2_{framing_mode}")


    d1, d2 = st.columns(2)

    # C3: Ablation (hapus fitur dominan)
    with d1:
        top_feature = imp.iloc[-1]["Fitur"] if len(imp) else None
        if top_feature is None or len(X_train.columns) <= 2:
            st.info("C3 tidak bisa: fitur terlalu sedikit.")
        else:
            cols_wo = [c for c in X_train.columns if c != top_feature]
            clf_wo = Pipeline(steps=[("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=3000))])
            clf_wo.fit(X_train[cols_wo], y_train)
            prob_wo = clf_wo.predict_proba(X_test[cols_wo])[:, 1]
            pred_wo = (prob_wo >= 0.5).astype(int)
            try:
                auc_wo = float(roc_auc_score(y_test, prob_wo))
            except Exception:
                auc_wo = np.nan

            perf = pd.DataFrame({
                "Model": ["Model penuh", f"Tanpa fitur: {top_feature}"],
                "AUC": [auc, auc_wo],
            })
            fig_c3 = px.bar(perf, x="Model", y="AUC", title="C3 — Ablation: Dampak Menghapus Fitur Dominan")
            fig_c3.update_traces(marker=dict(color=GREEN))
            fig_c3 = apply_support_style(fig_c3, supports_argument=(framing_mode != "Framing Data"))
            if framing_mode == "Framing Data" and np.isfinite(auc) and np.isfinite(auc_wo):
                y0, y1 = min(auc, auc_wo), max(auc, auc_wo)
                pad = (y1 - y0) * 0.25 if (y1 - y0) > 0 else 0.05
                fig_c3.update_yaxes(range=[max(0, y0 - pad), min(1, y1 + pad)])
            st.plotly_chart(fig_c3, use_container_width=True, key=f"bab3_c3_{framing_mode}")


    # C4: What-if income -> P75
    with d2:
        income_col = None
        for cand in ["income", "rincome", "realinc", "inc"]:
            c = _col_like(X_train, [cand])
            if c is not None:
                income_col = c
                break

        if income_col is None:
            st.info("C4 tidak bisa: kolom income tidak ditemukan pada fitur model.")
        else:
            Xw = X_test.copy()
            base_mean = float(np.mean(prob))

            p75 = float(np.nanpercentile(X_train[income_col].dropna(), 75))
            # What-if: naikkan income rendah ke minimal P75 (lebih realistis dari age)
            Xw[income_col] = np.where(
                Xw[income_col].isna(),
                Xw[income_col],
                np.maximum(Xw[income_col], p75)
            )

            prob_w = clf.predict_proba(Xw)[:, 1]
            after_mean = float(np.mean(prob_w))

            sim = pd.DataFrame({
                "Skenario": ["Asli", f"What-if: {income_col}→≥P75"],
                "Rata-rata Probabilitas": [base_mean, after_mean],
            })
            fig_c4 = px.bar(sim, x="Skenario", y="Rata-rata Probabilitas", title="C4 — What-if: Perbaikan Income Menurunkan Prediksi Risiko")
            fig_c4.update_traces(marker=dict(color=GREEN))
            fig_c4 = apply_support_style(fig_c4, supports_argument=(framing_mode != "Framing Data"))
        st.plotly_chart(fig_c4, use_container_width=True, key=f"bab3_c4_{framing_mode}")


    st.caption(f"Catatan: Prediktif saat ini memakai dataset: **{used}**.")


with tab4:
    st.markdown("### BAB IV — Pembongkaran Framing (Ethical Disclaimer + Versi Netral)")

    st.markdown("#### 1) Disclaimer Etika (Framing tanpa memalsukan data)")
    st.warning(
        "Bagian ini sengaja membongkar teknik framing yang dipakai di dashboard. "
        "Tujuannya edukasi literasi data: menunjukkan bahwa persepsi bisa berubah karena cara penyajian "
        "(rentang tahun, skala, subset, label, dan pilihan visual), bukan karena data dimanipulasi."
    )

    st.markdown("#### 2) Teknik Framing yang Dipakai di Dashboard (apa saja triknya)")
    st.markdown(
        """
            Berikut teknik yang **benar-benar muncul di implementasi**:

            1) **Windowing / seleksi rentang tahun**
            - Mode *Framing Data* memusatkan fokus ke rentang tertentu (contoh: 2019–2022) agar tren terlihat mendukung narasi.
            - Dampak: tren jadi tampak “lebih konsisten” atau “lebih mengarah” dibanding jika tahun diperluas.

            2) **Manipulasi skala (zoom)**
            - Beberapa grafik kepuasan dibatasi ke rentang **3–5** (atau range sempit).
            - Dampak: variasi ekstrem jadi “hilang”, sehingga terlihat stabil dan seolah “baik-baik saja”.

            3) **Menghilangkan konteks angka (menghapus tick label)**
            - Pada mode framing, sumbu-Y sering disembunyikan (`showticklabels=False`).
            - Dampak: audiens fokus ke bentuk garis/arah tren, bukan magnitude.

            4) **Sumbu-x “dipoles” (x tanpa gap)**
            - Fungsi `x_no_gap()` membuat sumbu-x palsu (0..n-1) tapi label tetap tahun.
            - Dampak: tampilan tren tampak lebih rapi dan “kontinu” meski jarak tahun aslinya tidak ditonjolkan.

            5) **Cherry-picking kategori/label**
            - Contoh: menghapus label tertentu seperti “lain-lain” / alasan tertentu di beberapa tampilan framing.
            - Dampak: komposisi alasan terlihat lebih “jelas” mendukung narasi.

            6) **Mencampur narasi lintas dataset (Indonesia vs affairs vs komentar YouTube)**
            - Data Indonesia menjelaskan nikah/cerai & alasan cerai (makro).
            - Data affairs menjelaskan kepuasan & status affair (mikro).
            - Sentimen YouTube menunjukkan opini publik (persepsi).
            - Dampak: cerita terasa utuh, padahal konteks populasi & definisi variabel berbeda.
        """
    )

    st.markdown("#### 3) Versi Netral (bagaimana seharusnya jika tidak framing)")
    st.markdown(
        """
            Jika tujuan presentasi adalah **objektif**, maka yang perlu dilakukan:

            - **Pakai rentang tahun penuh** (atau minimal jelaskan kenapa memilih window tertentu).
            - **Tampilkan sumbu dan tick label lengkap** agar pembaca tahu besaran angka.
            - **Hindari zoom sempit tanpa alasan**; jika zoom, tampilkan juga versi range penuh sebagai pembanding.
            - **Jangan menyembunyikan kategori besar** seperti “lain-lain” tanpa alasan metodologis (kalau dihapus, tulis alasan).
            - **Pisahkan kesimpulan per dataset**:
            - Indonesia (makro): tren nikah/cerai + alasan cerai
            - Affairs (mikro): hubungan affair–kepuasan (bukan “Indonesia”)
            - Sentimen: persepsi publik, bukan bukti kausal
        """
    )

    st.markdown("#### 4) Checklist ‘How to Lie with Statistics’ yang Terlihat (ringkas tapi jelas)")
    st.markdown(
        """
            ✅ **Selection bias (subset data)** → pilih tahun/kelompok tertentu untuk mendukung narasi  
            ✅ **Truncated/zoomed axis** → sempitkan range agar terlihat stabil  
            ✅ **Context removal** → sembunyikan tick label sehingga magnitude tidak terbaca  
            ✅ **Category manipulation** → buang kategori/label tertentu agar komposisi “lebih cantik”  
            ✅ **Cross-dataset inference** → buat narasi menyatu dari dataset berbeda konteks  
        """
    )



with tab5:
    st.markdown("### BAB V — Kesimpulan, Refleksi, dan Penutup")

    st.markdown("#### 1) Kesimpulan Utama (apa yang *terlihat* dari data)")
    st.markdown(
        """
        - **Perselingkuhan tidak otomatis “menghancurkan” pernikahan** jika indikator yang dipakai adalah *kepuasan pernikahan* pada dataset affairs: pada banyak kasus, kelompok *Affair* masih punya nilai kepuasan berada di rentang menengah–tinggi (terutama jika visualnya di-zoom).
        - **Perceraian (Indonesia) lebih dominan dijelaskan oleh alasan struktural** (mis. ekonomi/konflik/ditinggalkan) dibanding alasan moral tertentu, jika dilihat dari agregasi “alasan perceraian”.
        - **Opini publik cenderung negatif terhadap perselingkuhan**, terlihat dari kata/keyword negatif pada komentar (sentimen), sehingga terjadi kontras antara “narasi sosial” vs “indikator kepuasan pada dataset tertentu”.
        """
    )

    st.markdown("#### 2) Refleksi Kritis (kenapa framing bisa meyakinkan)")
    st.markdown(
        """
            Pada dashboard ini, saya menunjukkan bahwa *tanpa mengubah data mentah*, persepsi audiens bisa berubah lewat:
            - **Pemilihan rentang tahun / subset data** (mis. tahun tertentu yang mendukung narasi).
            - **Pengaturan skala (zoom y-axis)** sehingga variasi terlihat “lebih stabil” atau “lebih dramatis”.
            - **Penghilangan konteks angka (tick label)** agar fokus audiens pindah ke bentuk tren, bukan besarannya.
            - **Perbandingan lintas dataset** (Indonesia vs affairs) yang membuat cerita terasa menyatu, walau konteks sebenarnya berbeda.
        """
    )

    st.markdown("#### 3) Keterbatasan & Kejujuran Analisis (versi netral)")
    st.markdown(
        """
            - Dataset *affairs* bukan data Indonesia; ia menggambarkan populasi dan konteks berbeda → hasil tidak boleh digeneralisasi langsung ke fenomena nasional.
            - Kepuasan pernikahan ≠ keberlangsungan pernikahan; “tidak hancur” secara kepuasan belum tentu berarti “tidak berujung cerai”.
            - Data alasan cerai bersifat agregat; tidak bisa menyimpulkan sebab-akibat individual (hanya pola pelaporan kasus).
            - Analisis sentimen berbasis token sederhana → belum menangkap sarkasme, konteks kalimat, atau multi-makna kata.
        """
    )

    st.markdown("#### 4) Rekomendasi (apa yang seharusnya dilakukan pembaca data)")
    st.markdown(
        """
            - Selalu cek **skala, rentang, dan baseline** sebelum percaya kesimpulan dari grafik.
            - Pisahkan “temuan data” vs “narasi yang dibangun”.
            - Lihat *versi framing* dan *versi keadaan nyata* untuk membedakan “cerita” dan “kondisi”.
        """
    )
    st.divider()

    try:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rata-rata Nikah Nasional / Tahun", f"{avg_nikah:,.0f}" if np.isfinite(avg_nikah) else "-")
        col2.metric("Rata-rata Cerai Nasional / Tahun", f"{avg_cerai:,.0f}" if np.isfinite(avg_cerai) else "-")
        col3.metric("Rasio Cerai/Nikah", f"{ratio:.1f}%" if np.isfinite(ratio) else "-")
        col4.metric("Affair rate (dataset)", f"{affair_rate:.1f}%" if np.isfinite(affair_rate) else "-")
    except Exception:
        pass
