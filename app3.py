"""
=====================================================================
DYNAMIC EXCEL DASHBOARD - GENERIC AUTO-ANALYTICS ENGINE
=====================================================================
Dashboard generik berbasis Streamlit yang membaca file Excel apapun
(tanpa hardcode dataset tertentu), lalu otomatis mendeteksi struktur
data (numeric, date, year, month, category), membangun filter,
KPI, visualisasi, dan export hasil filter.

Cara jalankan:
    streamlit run app.py
=====================================================================
"""

# =====================================================================
# PART 1: IMPORTS, PAGE CONFIG, CUSTOM CSS, HELPER CONSTANTS
# =====================================================================

import io
import re
import calendar

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------------------------------------------------
# Page Config (harus dipanggil pertama sekali di script)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Excel Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom CSS agar dashboard terlihat compact & profesional
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 12px 14px 8px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        color: #555;
    }
    section[data-testid="stSidebar"] {
        background-color: #fafafa;
    }
    h1, h2, h3 {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 14px;
        border-radius: 8px 8px 0 0;
    }
    hr {
        margin: 0.6rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MONTH_ORDER = list(calendar.month_name)[1:]  # January..December
MONTH_ALIAS = {m.lower(): i + 1 for i, m in enumerate(MONTH_ORDER)}
MONTH_ALIAS.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTH_ORDER)})

CATEGORY_MAX_UNIQUE = 200
MAX_NUMERIC_SLIDERS = 20
MAX_CATEGORY_FILTERS = 20


# =====================================================================
# PART 2: DATA CLEANING & AUTO DETECTION ENGINE
# =====================================================================

def is_text_dtype(series: pd.Series) -> bool:
    """True untuk kolom teks, kompatibel dengan dtype 'object' klasik
    maupun dtype 'string'/StringDtype baru (pandas >= 2.x/3.x), tapi
    tetap False untuk kolom numeric/datetime/bool."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return False
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan nama kolom: strip spasi, hapus newline, rapikan."""
    df = df.copy()
    new_cols = []
    for c in df.columns:
        c = str(c).strip()
        c = re.sub(r"\s+", " ", c)
        c = c.replace("\n", " ").replace("\r", " ")
        new_cols.append(c)
    df.columns = new_cols
    return df


def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Hapus kolom yang seluruhnya kosong (NaN semua) atau unnamed kosong."""
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    for c in unnamed_cols:
        if df[c].isna().all():
            df = df.drop(columns=[c])
    return df


def clean_numeric_string(series: pd.Series, assume_single_separator_as_thousands: bool = False) -> pd.Series:
    """Bersihkan string numerik jadi angka PERSIS tanpa pembulatan.

    Kasus yang TIDAK ambigu (selalu otomatis diproses, aman):
    - Ada titik DAN koma sekaligus -> yang muncul PALING TERAKHIR dianggap desimal,
      sisanya dianggap pemisah ribuan. Contoh: "8.847,50" -> 8847.5, "8,847.50" -> 8847.5
    - Titik lebih dari 1 kali (tanpa koma) -> pasti pemisah ribuan. Contoh: "1.234.567" -> 1234567
    - Koma lebih dari 1 kali (tanpa titik) -> pasti pemisah ribuan. Contoh: "1,234,567" -> 1234567

    Kasus AMBIGU (cuma 1 titik ATAU 1 koma, contoh "8.847" atau "8,847") -
    DEFAULT diperlakukan sebagai DESIMAL ASLI (titik dibiarkan apa adanya,
    koma diganti jadi titik) supaya TIDAK ADA angka yang berubah tanpa
    sepengetahuan user. Kalau assume_single_separator_as_thousands=True,
    baru dicoba tebak sebagai pemisah ribuan (hanya kalau pola-nya persis
    3 digit di belakang simbol).
    """
    def _clean(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val
        s = str(val).strip()
        if s == "":
            return np.nan

        # tangani angka negatif dalam kurung, misal (1000)
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1]

        # hapus simbol mata uang & spasi (BUKAN koma/titik - itu dihandle terpisah di bawah)
        s = re.sub(r"[Rp$€£¥\s]", "", s)

        # tangani persen
        is_percent = s.endswith("%")
        s = s.replace("%", "")

        has_comma = "," in s
        has_dot = "." in s

        if has_comma and has_dot:
            # Tidak ambigu: yang terakhir muncul = desimal
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif has_dot and not has_comma:
            if s.count(".") > 1:
                # Tidak ambigu: banyak titik = pasti pemisah ribuan, contoh "1.234.567"
                s = s.replace(".", "")
            elif assume_single_separator_as_thousands:
                int_part, frac_part = s.split(".")
                if len(frac_part) == 3 and int_part not in ("", "0") and len(int_part) <= 3:
                    s = s.replace(".", "")
                # else dibiarkan sebagai desimal
            # default: dibiarkan apa adanya (desimal asli), contoh "8.847" tetap 8.847
        elif has_comma and not has_dot:
            if s.count(",") > 1:
                # Tidak ambigu: banyak koma = pasti pemisah ribuan
                s = s.replace(",", "")
            elif assume_single_separator_as_thousands:
                int_part, frac_part = s.split(",")
                if len(frac_part) == 3 and int_part not in ("", "0") and len(int_part) <= 3:
                    s = s.replace(",", "")
                else:
                    s = s.replace(",", ".")
            else:
                # default: koma tunggal dianggap desimal, contoh "8,5" -> 8.5
                s = s.replace(",", ".")

        try:
            num = float(s)
            if neg:
                num = -num
            if is_percent:
                num = num / 100
            return num
        except ValueError:
            return np.nan
    return series.apply(_clean)


def try_convert_numeric(df: pd.DataFrame, assume_dot_comma_as_thousands: bool = False) -> pd.DataFrame:
    """Coba konversi kolom object menjadi numeric jika mayoritas berhasil."""
    df = df.copy()
    for col in df.columns:
        if is_text_dtype(df[col]):
            cleaned = clean_numeric_string(df[col], assume_dot_comma_as_thousands)
            valid_ratio = cleaned.notna().sum() / max(len(cleaned), 1)
            original_non_null = df[col].notna().sum()
            if original_non_null > 0 and valid_ratio >= 0.7 and cleaned.notna().sum() > 0:
                df[col] = cleaned
    return df


def detect_date_columns(df: pd.DataFrame) -> list:
    """Deteksi kolom yang mayoritas isinya bisa dikonversi ke datetime."""
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            continue
        try:
            converted = pd.to_datetime(df[col], errors="coerce")
            valid_ratio = converted.notna().sum() / max(df[col].notna().sum(), 1)
            if valid_ratio >= 0.7 and converted.notna().sum() > 0:
                date_cols.append(col)
        except Exception:
            continue
    return date_cols


def detect_year_columns(df: pd.DataFrame) -> list:
    """Deteksi kolom tahun: nama mengandung 'year' ATAU isi angka 4 digit tahun wajar."""
    year_cols = []
    for col in df.columns:
        name_match = "year" in str(col).lower() or "tahun" in str(col).lower()
        value_match = False
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) > 0:
                in_range = vals.between(1990, 2100)
                value_match = in_range.mean() >= 0.8
        if name_match or value_match:
            year_cols.append(col)
    return year_cols


def detect_month_columns(df: pd.DataFrame) -> list:
    """Deteksi kolom bulan: nama mengandung 'month' ATAU isi nama/angka bulan."""
    month_cols = []
    for col in df.columns:
        name_match = "month" in str(col).lower() or "bulan" in str(col).lower()
        value_match = False
        sample = df[col].dropna().astype(str).str.lower().str.strip()
        if len(sample) > 0:
            matches = sample.isin(MONTH_ALIAS.keys())
            if matches.mean() >= 0.7:
                value_match = True
            elif pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna()
                if len(vals) > 0 and vals.between(1, 12).mean() >= 0.8:
                    value_match = True
        if name_match or value_match:
            month_cols.append(col)
    return month_cols


def month_to_number(val):
    """Konversi nilai bulan (nama atau angka) menjadi angka 1-12."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = int(val)
        return v if 1 <= v <= 12 else np.nan
    s = str(val).strip().lower()
    return MONTH_ALIAS.get(s, np.nan)


def detect_category_columns(df: pd.DataFrame, exclude: list) -> list:
    """Deteksi kolom kategorikal: object/string dengan unique value terbatas."""
    cat_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if is_text_dtype(df[col]) or str(df[col].dtype).startswith("category"):
            n_unique = df[col].nunique(dropna=True)
            if 0 < n_unique <= CATEGORY_MAX_UNIQUE:
                cat_cols.append(col)
    return cat_cols


ID_NAME_PATTERN = re.compile(
    r"(no\.?$|no\.?\s|number|nomor|\bref\b|reference|\bid\b|\bcode\b|\bkode\b)",
    re.IGNORECASE,
)

# Pola nama kolom yang KEMUNGKINAN BESAR cuma nomor urut/serial (bukan business
# key beneran seperti "Contract Ref.No" yang wajar berulang untuk kontrak sama).
# Sengaja dibuat SEMPIT (harus cocok nama secara keseluruhan) supaya nggak
# ketuker sama kolom identifier penting.
PURE_SERIAL_NAME_PATTERN = re.compile(
    r"^(no\.?|s\/n|sn|seq|sequence|index|row\s*no\.?|nomor\s*urut)$",
    re.IGNORECASE,
)


def detect_pure_serial_columns(df: pd.DataFrame) -> list:
    """Deteksi kolom nomor urut MURNI (misal 'No.' yang isinya 1,2,3,4,...)
    yang nilainya nyaris selalu unik per baris. Kolom kayak gini HARUS
    diabaikan saat cek duplikat baris, karena kalau tidak, baris yang
    sebenarnya sama persis (cuma beda nomor urut) tidak akan pernah
    terdeteksi sebagai duplikat -> total jadi salah / dobel hitung."""
    serial_cols = []
    for col in df.columns:
        name_clean = str(col).strip()
        if PURE_SERIAL_NAME_PATTERN.match(name_clean):
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            uniq_ratio = non_null.nunique() / len(non_null)
            if uniq_ratio > 0.9:
                serial_cols.append(col)
    return serial_cols


def detect_id_like_columns(df: pd.DataFrame, numeric_cols: list) -> list:
    """Deteksi kolom numeric yang sebenarnya ID/reference number (misal
    'Contract Ref.No', 'Shipment No.', 'No.') -> bukan angka yang bermakna
    untuk dijumlah/dirata-rata, tapi tetap berguna sebagai kunci breakdown."""
    id_cols = []
    for col in numeric_cols:
        if ID_NAME_PATTERN.search(str(col)):
            id_cols.append(col)
    return id_cols


def fmt_num(x, decimals: int = 2) -> str:
    """Format angka pakai pemisah ribuan, MAKSIMAL `decimals` digit di
    belakang koma - tapi kalau angkanya bulat/ga butuh desimal, desimalnya
    dibuang (misal 3000000 -> '3,000,000', BUKAN '3,000,000.00').
    Kalau ada pecahannya, tetap tampil sampai `decimals` digit
    (misal 1234.5 -> '1,234.50')."""
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return "-"
        rounded = round(float(x), decimals)
    except (TypeError, ValueError):
        return str(x)
    if rounded == int(rounded):
        return f"{int(rounded):,}"
    return f"{rounded:,.{decimals}f}"


def detect_column_unit(col_name: str) -> str:
    """Auto-detect keterangan satuan dari NAMA KOLOM di file Excel, misal:
    - 'Volume (Million MT)'  -> 'Million MT'
    - 'Weight in Kg'         -> 'Kg'
    - 'Sales (USD Thousand)' -> 'USD Thousand'
    - 'Revenue in Million USD' -> 'Million USD'
    Ini bikin dashboard tetap universal (ga hardcode nama kolom tertentu) -
    keterangan satuan otomatis ngikut apa yang ditulis di header Excel-nya.
    Return string kosong "" kalau ga ketemu pola satuan apapun."""
    if not isinstance(col_name, str):
        return ""

    # 1) Pola dalam kurung di akhir nama kolom: "Nama Kolom (satuan)"
    paren_matches = re.findall(r"\(([^)]+)\)", col_name)
    for candidate in reversed(paren_matches):
        candidate = candidate.strip()
        # satuan harus mengandung huruf (bukan cuma kode/angka ga jelas)
        # dan cukup pendek supaya bukan salah tangkap kalimat panjang
        if re.search(r"[A-Za-z]", candidate) and len(candidate) <= 40:
            return candidate

    # 2) Pola "in <satuan>" di akhir nama kolom, misal "Revenue in Million USD"
    in_match = re.search(r"\bin\s+([A-Za-z][A-Za-z0-9\s./%-]{1,35})$", col_name, re.IGNORECASE)
    if in_match:
        return in_match.group(1).strip()

    return ""


def show_unit_note(col_name: str) -> None:
    """Tampilkan caption kecil kalau nama kolom `col_name` punya keterangan
    satuan yang bisa dideteksi otomatis (lihat detect_column_unit)."""
    unit = detect_column_unit(col_name)
    if unit:
        st.caption(f"📏 Satuan terdeteksi dari nama kolom **{col_name}**: *in {unit}*")


# Kata skala yang dikenali (Indonesia & Inggris) beserta pengalinya.
SCALE_WORDS = {
    "thousand": 1_000, "ribu": 1_000,
    "million": 1_000_000, "juta": 1_000_000,
    "billion": 1_000_000_000, "miliar": 1_000_000_000, "milyar": 1_000_000_000,
}


def parse_unit_parts(unit_text: str):
    """Pecah keterangan satuan jadi (skala_eksplisit, kata_skala, satuan_dasar).
    Misal: 'Million MT' -> (1_000_000, 'Million', 'MT')
           'MT'         -> (None, None, 'MT')
           'USD Thousand' -> (1_000, 'Thousand', 'USD')
    Kalau ga ada kata skala di teksnya, skala_eksplisit = None (artinya
    data MASIH angka mentah/asli, belum di-scale)."""
    if not unit_text:
        return None, None, ""
    tokens = unit_text.split()
    for i, tok in enumerate(tokens):
        key = tok.lower()
        if key in SCALE_WORDS:
            rest = " ".join(tokens[:i] + tokens[i + 1:]).strip()
            return SCALE_WORDS[key], tok, rest
    return None, None, unit_text


def auto_scale_factor(max_abs_value: float):
    """Auto-detect pembagi (Thousand/Million/Billion) berdasarkan besarnya
    angka, supaya angka yang ditampilkan di chart lebih ringkas & gampang
    dibaca (misal 22.000.123 jadi 22.000 dengan keterangan 'In Thousand').
    Angka kecil (< 1.000) ga di-scale sama sekali."""
    if max_abs_value is None or pd.isna(max_abs_value) or max_abs_value == 0:
        return 1, ""
    max_abs_value = abs(max_abs_value)
    if max_abs_value >= 1_000_000_000:
        return 1_000_000_000, "Billion"
    if max_abs_value >= 1_000_000:
        return 1_000_000, "Million"
    if max_abs_value >= 1_000:
        return 1_000, "Thousand"
    return 1, ""


def get_display_scale(col_name: str, series: pd.Series):
    """Tentukan (divisor, caption) buat nampilin angka kolom `col_name`
    dengan data `series` di chart:
    - Kalau nama kolom SUDAH eksplisit nyebut skala (misal 'Volume (Million MT)'),
      dianggap datanya SUDAH dalam skala itu -> ga dibagi lagi (divisor=1),
      caption langsung dari situ: 'In Million MT'.
    - Kalau BELUM ada skala eksplisit (misal cuma 'Volume (MT)' atau 'Volume'
      dengan angka mentah besar seperti 22.000.123), auto-detect skala yang
      pas dari besarnya angka, angkanya DIBAGI biar ringkas, caption jadi
      misal 'In Thousand MT' - inilah yang bikin '22.000.123' -> '22.000'."""
    unit_text = detect_column_unit(col_name)
    explicit_scale_val, explicit_scale_word, base_unit = parse_unit_parts(unit_text)

    if explicit_scale_val:
        caption = f"In {explicit_scale_word} {base_unit}".strip()
        return 1, caption

    max_abs = series.abs().max() if series is not None and len(series) else None
    auto_div, auto_label = auto_scale_factor(max_abs)
    if auto_div == 1:
        return 1, (f"In {base_unit}" if base_unit else "")

    caption = f"In {auto_label}" + (f" {base_unit}" if base_unit else "")
    return auto_div, caption


def scale_for_display(data: pd.DataFrame, value_col: str, unit_source_col: str):
    """Terapkan auto-scale ke `data[value_col]` berdasarkan nama kolom asal
    `unit_source_col` (dipakai buat deteksi satuan). Return (data_baru_yang_
    sudah_di-scale, caption_satuan). Kalau `unit_source_col` None/kosong
    (misal lagi hitung Jumlah Baris, bukan kolom angka beneran), ga di-scale."""
    if not unit_source_col or value_col not in data.columns or data.empty:
        return data, ""
    divisor, caption = get_display_scale(unit_source_col, data[value_col])
    if divisor == 1:
        return data, caption
    scaled = data.copy()
    scaled[value_col] = scaled[value_col] / divisor
    return scaled, caption


def group_selected_as_others(
    data: pd.DataFrame, label_col: str, value_col: str, selected_values: list
) -> pd.DataFrame:
    """Gabungkan baris-baris yang labelnya ada di `selected_values` jadi satu
    baris 'Others' - user yang pilih sendiri kategori mana yang mau dianggap
    'Others', bukan otomatis berdasarkan persentase."""
    if data.empty or not selected_values:
        return data
    is_selected = data[label_col].astype(str).isin(selected_values)
    if is_selected.sum() == 0:
        return data
    big_part = data[~is_selected]
    small_part = data[is_selected]
    others_total = small_part[value_col].sum()
    others_row = pd.DataFrame({label_col: ["Others"], value_col: [others_total]})
    result = pd.concat([big_part, others_row], ignore_index=True)
    return result.sort_values(value_col, ascending=False).reset_index(drop=True)


def show_chart(
    fig, n_categories: int = None, is_pie: bool = False, base_height: int = 460, unit_caption: str = ""
) -> None:
    """Tampilkan plotly chart dengan styling konsisten supaya:
    1. Font lebih besar & tetap besar walau di-zoom/fullscreen (di-set ke
       ukuran fix yang cukup besar, bukan ngikut ukuran container).
    2. Label/teks (nama kategori, angka, legend) TIDAK KEPOTONG - pakai
       automargin, margin lebih lega, dan tinggi chart otomatis nambah
       kalau kategorinya banyak.
    3. Kalau di-download (tombol kamera di pojok kanan atas chart), hasil
       gambarnya beresolusi besar & tajam (bukan ukuran kecil yang bikin
       tulisan kepotong/pecah).
    `n_categories`: jumlah kategori/slice/bar, dipakai buat nentuin tinggi
    chart & apakah label sumbu-X perlu dimiringkan biar ga tabrakan.
    `unit_caption`: kalau diisi (misal 'In Thousand MT'), ditampilkan
    sebagai anotasi kecil di pojok kiri atas chart - persis kayak
    keterangan satuan di pojok chart pada umumnya."""
    fig.update_layout(
        font=dict(size=15),
        title_font=dict(size=19),
        legend=dict(font=dict(size=13)),
        margin=dict(l=70, r=50, t=70, b=90),
        uniformtext_minsize=12,
        uniformtext_mode="show",
    )
    fig.update_xaxes(automargin=True, tickfont=dict(size=13))
    fig.update_yaxes(automargin=True, tickfont=dict(size=13))

    height = base_height
    if n_categories:
        # makin banyak kategori, chart makin tinggi supaya label & legend
        # ga numpuk/kepotong
        height = max(base_height, base_height + (n_categories - 6) * 22)
        if n_categories > 8 and not is_pie:
            fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=height)

    if is_pie:
        fig.update_traces(textfont=dict(size=13))
        # Ini KUNCI-nya buat label outside pie yang kepotong: automargin
        # bikin plotly otomatis melebarkan margin figure supaya teks label
        # di luar slice (nama kategori + persen + angka) selalu muat, ga
        # kepotong di tepi chart.
        fig.update_traces(automargin=True, selector=dict(type="pie"))
        fig.update_layout(margin=dict(l=90, r=90, t=90, b=90))

    if unit_caption:
        fig.add_annotation(
            text=unit_caption,
            xref="paper", yref="paper",
            x=0, y=1.12, xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=13, color="#555555"),
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            # Ukuran & skala gambar hasil download dibesarkan supaya tulisan
            # tetap jelas terbaca, ga kepotong/pecah kayak sebelumnya.
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chart",
                "height": max(900, height * 2),
                "width": 1600,
                "scale": 3,
            },
        },
    )


def render_full_vs_others_pies(
    data: pd.DataFrame, label_col: str, value_col: str, title_prefix: str, others_selected: list,
    unit_source_col: str = None,
) -> None:
    """Tampilkan 2 pie chart perbandingan: kiri/atas versi FULL (semua
    kategori apa adanya), kanan/bawah versi dengan kategori terpilih
    digabung jadi 'Others'. Kalau kategorinya banyak, ditumpuk VERTIKAL
    (full width) bukan berdampingan, supaya label outside-nya ada ruang
    lebih lega dan ga kepotong.
    `unit_source_col`: nama kolom numeric ASLI (buat deteksi satuan &
    auto-scale angka gede, misal 22.000.123 -> 22.000 + caption
    'In Thousand MT'). Isi None kalau value_col bukan kolom angka asli
    (misal lagi hitung Jumlah Baris)."""
    data, unit_caption = scale_for_display(data, value_col, unit_source_col)
    n_max = max(len(data), len(group_selected_as_others(data, label_col, value_col, others_selected)))
    stack_vertically = n_max > 10

    if stack_vertically:
        pc_full = pc_others = st.container()
    else:
        pc_full, pc_others = st.columns(2)

    with pc_full:
        st.caption("📊 Full (semua kategori)")
        fig_full = px.pie(data, names=label_col, values=value_col, title=f"{title_prefix} - Full")
        fig_full.update_traces(
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
            hovertemplate="%{label}<br>Persentase: %{percent:.2%}<br>Nilai: %{value:,.2~f}<extra></extra>",
        )
        show_chart(fig_full, n_categories=len(data), is_pie=True, unit_caption=unit_caption)
    with pc_others:
        st.caption("🗂️ Dengan 'Others'")
        data_with_others = group_selected_as_others(data, label_col, value_col, others_selected)
        fig_others = px.pie(data_with_others, names=label_col, values=value_col, title=f"{title_prefix} - Others")
        fig_others.update_traces(
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
            hovertemplate="%{label}<br>Persentase: %{percent:.2%}<br>Nilai: %{value:,.2~f}<extra></extra>",
        )
        show_chart(fig_others, n_categories=len(data_with_others), is_pie=True, unit_caption=unit_caption)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Konversi dataframe menjadi bytes file Excel untuk diunduh."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered Data")
        workbook = writer.book
        worksheet = writer.sheets["Filtered Data"]
        header_format = workbook.add_format({
            "bold": True, "bg_color": "#4472C4", "font_color": "white", "border": 1
        })
        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_format)
            max_width = max(len(str(col_name)), 12)
            worksheet.set_column(col_num, col_num, max_width + 4)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def guess_header_row(file_bytes: bytes, sheet_name, max_scan: int = 15) -> int:
    """Tebak baris header dengan heuristik: baris dengan jumlah sel terisi
    terbanyak di antara beberapa baris pertama biasanya adalah baris header
    (baris judul/kosong di atasnya biasanya cuma terisi sebagian)."""
    try:
        preview = pd.read_excel(
            io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=max_scan
        )
    except Exception:
        return 0
    if preview.empty:
        return 0
    non_null_counts = preview.notna().sum(axis=1)
    return int(non_null_counts.idxmax())


def rename_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom yang tetap tidak punya nama (header aslinya memang kosong)
    diberi nama yang lebih jelas daripada 'Unnamed: N' bawaan pandas."""
    df = df.copy()
    new_cols = []
    counter = 1
    for c in df.columns:
        if str(c).lower().startswith("unnamed"):
            new_cols.append(f"Kolom Tanpa Nama {counter}")
            counter += 1
        else:
            new_cols.append(c)
    df.columns = new_cols
    return df


@st.cache_data(show_spinner=False)
def load_and_process(file_bytes: bytes, sheet_name, header_row: int = 0,
                      assume_dot_comma_as_thousands: bool = False):
    """Load Excel dari bytes, bersihkan, dan jalankan auto-detection engine."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=header_row)

    df = clean_column_names(df)
    df = drop_empty_columns(df)
    df = rename_unnamed_columns(df)

    # PENTING: sebelum cek duplikat, kolom nomor urut murni (misal "No.")
    # DIABAIKAN dari perbandingan. Kalau tidak, baris yang sebenarnya
    # transaksi sama persis (cuma beda nomor urut) tidak akan pernah
    # kedeteksi sebagai duplikat, dan ikut ke-hitung dobel di semua
    # Total/Summary.
    serial_cols = detect_pure_serial_columns(df)
    dedup_subset = [c for c in df.columns if c not in serial_cols]
    df = df.drop_duplicates(subset=dedup_subset if dedup_subset else None)

    df = try_convert_numeric(df, assume_dot_comma_as_thousands)

    date_cols = detect_date_columns(df)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    year_cols = detect_year_columns(df)
    month_cols = detect_month_columns(df)

    # Jika hanya ada date column (tidak ada year/month eksplisit),
    # ekstrak otomatis Year, Month, Month Name dari date pertama.
    derived_year_col, derived_month_col, derived_monthname_col = None, None, None
    if date_cols and not year_cols and not month_cols:
        base_date = date_cols[0]
        derived_year_col = f"{base_date} - Year"
        derived_month_col = f"{base_date} - Month"
        derived_monthname_col = f"{base_date} - Month Name"
        df[derived_year_col] = df[base_date].dt.year
        df[derived_month_col] = df[base_date].dt.month
        df[derived_monthname_col] = df[base_date].dt.month_name()
        year_cols = [derived_year_col]
        month_cols = [derived_monthname_col]

    numeric_cols_raw = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                        and c not in year_cols]

    # Pisahkan kolom ID/reference number (misal "Contract Ref.No", "Shipment No.")
    # dari kolom numeric yang benar-benar bermakna untuk dijumlah/dirata-rata.
    id_like_cols = detect_id_like_columns(df, numeric_cols_raw)
    numeric_cols = [c for c in numeric_cols_raw if c not in id_like_cols]

    exclude_from_category = set(date_cols) | set(year_cols) | set(month_cols) | set(numeric_cols)
    category_cols = detect_category_columns(df, exclude=list(exclude_from_category))
    # id_like_cols SENGAJA tidak digabung ke category_cols di sini, supaya
    # tidak membebani sidebar Category Filter dengan ratusan checkbox
    # (misal Contract Ref.No). id_like_cols tetap disimpan terpisah di meta
    # untuk dipakai khusus di dropdown breakdown pada section Summary.

    meta = {
        "date_cols": date_cols,
        "year_cols": year_cols,
        "month_cols": month_cols,
        "numeric_cols": numeric_cols,
        "category_cols": category_cols,
        "id_like_cols": id_like_cols,
        "serial_cols": serial_cols,
        "derived_month_col": derived_month_col,
    }
    return df, meta


# =====================================================================
# PART 3: SIDEBAR - UPLOAD & SMART FILTER
# =====================================================================

st.title("📊 Dynamic Excel Dashboard")
st.caption(
    "Upload file Excel apa pun — dashboard akan otomatis mendeteksi struktur "
    "data, membangun filter, KPI, dan visualisasi yang relevan."
)

with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("👈 Silakan upload file Excel (.xlsx) di sidebar untuk memulai analisis.")
    st.stop()

file_bytes = uploaded_file.getvalue()

# Pilih sheet jika file punya banyak sheet
try:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_names = xls.sheet_names
except Exception as e:
    st.error(f"Gagal membaca file Excel: {e}")
    st.stop()

with st.sidebar:
    if len(sheet_names) > 1:
        selected_sheet = st.selectbox("Pilih Sheet", sheet_names, index=0)
    else:
        selected_sheet = sheet_names[0]

# -----------------------------------------------------------------
# DETEKSI BARIS HEADER: file real-world sering punya baris judul/kosong
# di atas baris header asli, yang bikin pandas gagal ngenalin nama kolom
# (muncul jadi "Unnamed: N") dan data ikut geser -> total ikut salah.
# Kasih preview mentah + biarin user koreksi baris header kalau tebakan
# otomatis meleset.
# -----------------------------------------------------------------
guessed_header_row = guess_header_row(file_bytes, selected_sheet)

with st.expander("🧭 Cek & Atur Baris Header (kalau ada kolom 'Unnamed' / total salah)", expanded=False):
    st.caption(
        "Preview mentah 10 baris pertama dari file (belum diproses). "
        "Pilih baris mana yang berisi NAMA KOLOM asli."
    )
    raw_preview = pd.read_excel(
        io.BytesIO(file_bytes), sheet_name=selected_sheet, header=None, nrows=10
    )
    st.dataframe(raw_preview, use_container_width=True)
    header_row = st.number_input(
        "Baris ke berapa (mulai dari 0) yang jadi header?",
        min_value=0, max_value=9, value=guessed_header_row, step=1,
    )
    st.markdown("---")
    st.caption(
        "Kalau ada kolom angka berupa TEKS yang pakai titik/koma tunggal dan ambigu "
        "(contoh: \"8.847\"), defaultnya dianggap DESIMAL ASLI (8.847 tetap 8.847) "
        "supaya tidak ada angka yang berubah tanpa kamu sadari. Aktifkan opsi di bawah "
        "HANYA kalau kamu yakin titik/koma tunggal itu maksudnya pemisah ribuan."
    )
    assume_dot_comma_as_thousands = st.checkbox(
        "Anggap titik/koma TUNGGAL pada angka teks sebagai pemisah ribuan "
        "(misal \"8.847\" -> 8847)",
        value=False,
    )

try:
    df_raw, meta = load_and_process(
        file_bytes, selected_sheet, header_row, assume_dot_comma_as_thousands
    )
except Exception as e:
    st.error(f"Gagal memproses data: {e}")
    st.stop()

if df_raw.empty:
    st.warning("File Excel berhasil dibaca tetapi tidak ada data yang bisa ditampilkan.")
    st.stop()

date_cols = meta["date_cols"]
year_cols = meta["year_cols"]
month_cols = meta["month_cols"]
numeric_cols = meta["numeric_cols"]
category_cols = meta["category_cols"]
id_like_cols = meta["id_like_cols"]
serial_cols = meta["serial_cols"]

# -----------------------------------------------------------------
# SEMBUNYIKAN KOLOM: biar kolom yang tidak relevan (misal Laycan, ETA,
# atau kolom internal lain) bisa dibuang dari seluruh dashboard tanpa
# perlu edit file Excel sumbernya. Ini generic - berlaku untuk kolom
# apapun, tidak hardcode nama kolom tertentu.
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    hidden_cols = st.multiselect(
        "🚫 Sembunyikan Kolom (opsional)",
        options=list(df_raw.columns),
        default=[],
        help="Kolom yang dipilih di sini akan dihilangkan dari seluruh dashboard: "
             "filter, KPI, chart, dan data preview.",
    )

if hidden_cols:
    df_raw = df_raw.drop(columns=[c for c in hidden_cols if c in df_raw.columns])
    date_cols = [c for c in date_cols if c not in hidden_cols]
    year_cols = [c for c in year_cols if c not in hidden_cols]
    month_cols = [c for c in month_cols if c not in hidden_cols]
    numeric_cols = [c for c in numeric_cols if c not in hidden_cols]
    category_cols = [c for c in category_cols if c not in hidden_cols]
    id_like_cols = [c for c in id_like_cols if c not in hidden_cols]

# -----------------------------------------------------------------
# Panel debug: tunjukkan hasil deteksi kolom apa adanya, supaya user
# bisa cek langsung kalau ada kolom yang tidak terbaca sesuai harapan.
# -----------------------------------------------------------------
all_detected = set(date_cols) | set(year_cols) | set(month_cols) | set(numeric_cols) | set(category_cols)
undetected_cols = [c for c in df_raw.columns if c not in all_detected]

with st.expander("🔍 Kolom Terdeteksi dari File Ini (klik untuk cek)", expanded=False):
    if serial_cols:
        st.info(
            f"ℹ️ Kolom nomor urut murni terdeteksi: **{', '.join(serial_cols)}** — "
            f"kolom ini DIABAIKAN saat cek baris duplikat, supaya baris yang datanya "
            f"sama persis (cuma beda nomor urut) tetap kehitung sebagai duplikat."
        )
    d1, d2, d3 = st.columns(3)
    d1.markdown("**📅 Date Columns**")
    d1.write(date_cols if date_cols else "-")
    d1.markdown("**📆 Year Columns**")
    d1.write(year_cols if year_cols else "-")
    d1.markdown("**🗓️ Month Columns**")
    d1.write(month_cols if month_cols else "-")
    d2.markdown("**🔢 Numeric Columns** (dijumlah/dirata-rata)")
    d2.write(numeric_cols if numeric_cols else "-")
    d2.markdown("**🆔 ID / Reference Columns** (numeric tapi bukan buat ditotal)")
    d2.write(id_like_cols if id_like_cols else "-")
    d3.markdown("**🏷️ Category Columns**")
    d3.write(category_cols if category_cols else "-")
    if undetected_cols:
        st.markdown("**⚠️ Belum masuk kategori manapun** (kemungkinan unique value terlalu banyak, atau tipe data campur):")
        st.write(undetected_cols)

# -----------------------------------------------------------------
# KECUALIKAN NILAI (EXCLUDE VALUES): user bisa buang nilai spesifik dari
# kolom manapun supaya nggak ikut dihitung di seluruh dashboard (KPI,
# Summary, chart, Data Preview, semuanya). Beda dari "Kelompokkan Nilai"
# yang MENGGABUNG - ini MEMBUANG total. Generic, berlaku untuk kolom &
# nilai apapun sesuai isi file.
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    with st.expander("🚫 Kecualikan Nilai dari Tabel (opsional)", expanded=False):
        st.caption(
            "Pilih nilai spesifik yang mau DIBUANG TOTAL dari seluruh dashboard "
            "(tidak ikut dihitung di KPI, Summary, chart, atau Data Preview manapun)."
        )
        exclude_target_cols = st.multiselect(
            "Kolom yang mau diatur pengecualiannya",
            options=list(df_raw.columns),
            key="exclude_target_cols",
        )

        exclude_rules = {}
        for ecol in exclude_target_cols:
            ecol_options = sorted(df_raw[ecol].dropna().astype(str).unique().tolist())
            selected_exclude_vals = st.multiselect(
                f"Nilai yang dikecualikan pada '{ecol}'",
                options=ecol_options,
                key=f"exclude_values_{ecol}",
            )
            if selected_exclude_vals:
                exclude_rules[ecol] = selected_exclude_vals

total_excluded_rows = 0
if exclude_rules:
    mask_exclude_total = pd.Series(False, index=df_raw.index)
    for ecol, evals in exclude_rules.items():
        mask_exclude_total = mask_exclude_total | df_raw[ecol].astype(str).isin(evals)
    total_excluded_rows = int(mask_exclude_total.sum())
    df_raw = df_raw[~mask_exclude_total].reset_index(drop=True)

if total_excluded_rows > 0:
    st.sidebar.success(f"✅ {total_excluded_rows:,} baris dikecualikan dari tabel.")

# -----------------------------------------------------------------
# KELOMPOKKAN NILAI (CUSTOM GROUPING): user bisa gabungin beberapa nilai
# dari 1 kolom kategori jadi 1 grup baru (misal nilai "A", "B", "C"
# digabung jadi 1 grup baru "ABC"), sementara nilai lain yang tidak
# dipilih tetap apa adanya. Hasilnya bisa disimpan sebagai KOLOM BARU
# terpisah, ATAU digabung LANGSUNG ke kolom lain yang sudah ada (misal
# hasil grouping dianggap "Indonesia" lalu ditaruh di kolom "Country",
# berdampingan dengan nilai negara lain yang sudah ada). Generic -
# berlaku untuk kolom & nilai apapun sesuai isi file yang diupload.
# -----------------------------------------------------------------
if "custom_groups" not in st.session_state:
    st.session_state.custom_groups = {}  # {source_col: [{"name":..., "values":[...], "target_col": str|None}]}


def get_merge_groups_for_target(target_col: str) -> list:
    """Kembalikan semua grup (dari kolom sumber manapun) yang HASIL gabungannya
    ditimpakan ke `target_col` tertentu, contoh: grup 'Jakarta' (dari kolom
    'Kota', isi asli abc/abcd/abcde) yang digabungkan ke kolom 'Country'.
    Dipakai untuk fitur drill-down: begitu user pilih target_col yang punya
    nilai hasil gabungan, kita bisa tampilkan rincian isi aslinya."""
    result = []
    for src_col, groups in st.session_state.custom_groups.items():
        for g in groups:
            if g.get("target_col") == target_col:
                result.append({"name": g["name"], "values": g["values"], "src_col": src_col})
    return result

with st.sidebar:
    st.markdown("---")
    with st.expander("🗂️ Kelompokkan Nilai (opsional)", expanded=False):
        st.caption(
            "Gabungkan beberapa nilai jadi 1 grup baru sesuai kebutuhan kamu. "
            "Hasilnya bisa disimpan sebagai kolom baru terpisah, atau langsung "
            "digabung ke kolom lain yang sudah ada supaya nilainya berdampingan "
            "dengan nilai-nilai lain di kolom itu."
        )
        group_source_col = st.selectbox(
            "Kolom SUMBER (yang mau dikelompokkan ulang)",
            options=["(Tidak dipakai)"] + category_cols,
            key="group_source_col",
        )

        if group_source_col != "(Tidak dipakai)":
            if group_source_col not in st.session_state.custom_groups:
                st.session_state.custom_groups[group_source_col] = []

            all_groups_this_source = st.session_state.custom_groups[group_source_col]

            if all_groups_this_source:
                st.markdown("**Grup yang sudah dibuat:**")
                for i, g in enumerate(all_groups_this_source):
                    target_desc = "→ kolom baru" if not g.get("target_col") else f"→ digabung ke '{g['target_col']}'"
                    gcol1, gcol2 = st.columns([4, 1])
                    gcol1.write(f"🔸 **{g['name']}** {target_desc}: {', '.join(g['values'])}")
                    if gcol2.button("🗑️", key=f"del_group_{group_source_col}_{i}"):
                        st.session_state.custom_groups[group_source_col].pop(i)
                        st.rerun()

            st.markdown("**➕ Buat grup baru:**")

            other_cols_for_target = [c for c in category_cols if c != group_source_col]
            target_options = ["Simpan sebagai kolom baru khusus"] + [
                f"{c}  (gabungkan ke kolom ini)" for c in other_cols_for_target
            ]
            target_choice = st.selectbox(
                "Hasil grup ini mau ditaruh di mana?",
                options=target_options,
                key="new_group_target_choice",
                help="'Kolom baru' bikin kolom terpisah khusus grouping ini. "
                     "Pilih nama kolom lain untuk menimpa nilai di kolom itu "
                     "(cuma untuk baris yang cocok - baris lain di kolom itu tetap apa adanya).",
            )
            target_col_value = (
                None if target_choice == "Simpan sebagai kolom baru khusus"
                else target_choice.replace("  (gabungkan ke kolom ini)", "")
            )

            # Nilai dianggap "sudah dipakai" HANYA dibanding grup lain yang tujuannya
            # SAMA PERSIS (target_col sama) - supaya 1 nilai (misal "Jakarta") boleh
            # dipakai di grup "Indonesia" (target: Country) SEKALIGUS dipakai lagi
            # buat bikin sub-grup di dalam Indonesia (target: kolom baru lain).
            groups_same_target = [g for g in all_groups_this_source if g.get("target_col") == target_col_value]
            already_grouped_for_target = set()
            for g in groups_same_target:
                already_grouped_for_target.update(g["values"])

            all_source_values = sorted(df_raw[group_source_col].dropna().astype(str).unique().tolist())
            available_values = [v for v in all_source_values if v not in already_grouped_for_target]

            if available_values:
                new_group_name = st.text_input("Nama grup baru (misal 'Indonesia')", key="new_group_name_input")
                new_group_values = st.multiselect(
                    "Pilih nilai yang mau digabung ke grup ini",
                    options=available_values,
                    key="new_group_values_input",
                )
                if st.button("➕ Tambah Grup", key="add_group_btn"):
                    if new_group_name.strip() and new_group_values:
                        st.session_state.custom_groups[group_source_col].append(
                            {
                                "name": new_group_name.strip(),
                                "values": new_group_values,
                                "target_col": target_col_value,
                            }
                        )
                        st.rerun()
                    else:
                        st.warning("Isi nama grup dan pilih minimal 1 nilai dulu.")
            else:
                st.caption("Semua nilai yang relevan sudah masuk ke suatu grup untuk tujuan ini.")

# Terapkan grouping ke df_raw. Dua mode:
# 1) target_col kosong -> bikin KOLOM BARU "{src_col} (Dikelompokkan)"
# 2) target_col diisi  -> TIMPA nilai di kolom target itu (cuma untuk baris
#    yang cocok), baris lain di kolom target tetap nilai aslinya.
for src_col, groups in st.session_state.custom_groups.items():
    if not groups or src_col not in df_raw.columns:
        continue

    new_col_groups = [g for g in groups if not g.get("target_col")]
    merge_groups = [g for g in groups if g.get("target_col")]

    if new_col_groups:
        mapping = {}
        for g in new_col_groups:
            for v in g["values"]:
                mapping[v] = g["name"]
        new_col_name = f"{src_col} (Dikelompokkan)"
        df_raw[new_col_name] = df_raw[src_col].astype(str).map(mapping).fillna(df_raw[src_col].astype(str))
        if new_col_name not in category_cols:
            category_cols.append(new_col_name)

    for g in merge_groups:
        target_col = g["target_col"]
        if target_col not in df_raw.columns:
            continue
        mask_merge = df_raw[src_col].astype(str).isin(g["values"])
        df_raw.loc[mask_merge, target_col] = g["name"]

df_filtered = df_raw.copy()

with st.sidebar:
    st.markdown("---")
    st.header("🔎 Filter")

    # -----------------------------------------------------------------
    # A. TIME FILTER - Year
    # -----------------------------------------------------------------
    if year_cols:
        year_col = year_cols[0]
        year_values = sorted(df_raw[year_col].dropna().unique().tolist())
        year_values = [int(y) for y in year_values]
        if year_values:
            with st.expander("📅 Year Filter", expanded=True):
                selected_years = st.multiselect(
                    "Pilih Tahun", options=year_values, default=year_values
                )
            if selected_years:
                df_filtered = df_filtered[df_filtered[year_col].isin(selected_years)]

    # -----------------------------------------------------------------
    # A. TIME FILTER - Month
    # -----------------------------------------------------------------
    if month_cols:
        month_col = month_cols[0]
        raw_month_values = df_raw[month_col].dropna().unique().tolist()

        # Normalisasi ke nama bulan standar untuk urutan yang benar
        def _to_month_name(v):
            num = month_to_number(v)
            if pd.isna(num):
                return None
            return MONTH_ORDER[int(num) - 1]

        month_name_map = {v: _to_month_name(v) for v in raw_month_values}
        ordered_months = sorted(
            set(m for m in month_name_map.values() if m is not None),
            key=lambda m: MONTH_ORDER.index(m),
        )
        if ordered_months:
            with st.expander("🗓️ Month Filter", expanded=True):
                selected_months = st.multiselect(
                    "Pilih Bulan", options=ordered_months, default=ordered_months
                )
            if selected_months:
                allowed_raw_vals = [
                    raw for raw, name in month_name_map.items() if name in selected_months
                ]
                df_filtered = df_filtered[df_filtered[month_col].isin(allowed_raw_vals)]

    # -----------------------------------------------------------------
    # A. TIME FILTER - Date Range
    # -----------------------------------------------------------------
    if date_cols:
        date_col = date_cols[0]
        valid_dates = df_raw[date_col].dropna()
        if not valid_dates.empty:
            min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
            with st.expander("📆 Date Range Filter", expanded=False):
                date_range = st.date_input(
                    "Rentang Tanggal", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date,
                )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_d, end_d = date_range
                mask = (df_filtered[date_col].dt.date >= start_d) & \
                       (df_filtered[date_col].dt.date <= end_d)
                df_filtered = df_filtered[mask | df_filtered[date_col].isna()]

    # -----------------------------------------------------------------
    # B. CATEGORY FILTER
    # -----------------------------------------------------------------
    if category_cols:
        with st.expander("🏷️ Category Filter", expanded=True):
            for col in category_cols[:MAX_CATEGORY_FILTERS]:
                options = sorted(df_raw[col].dropna().astype(str).unique().tolist())
                if len(options) == 0:
                    continue
                selected_vals = st.multiselect(f"{col}", options=options, default=options)
                if selected_vals and len(selected_vals) < len(options):
                    df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected_vals)]

    # -----------------------------------------------------------------
    # C. NUMERIC FILTER
    # -----------------------------------------------------------------
    if numeric_cols:
        with st.expander("🔢 Numeric Filter", expanded=True):
            for col in numeric_cols[:MAX_NUMERIC_SLIDERS]:
                col_data = df_raw[col].dropna()
                if col_data.empty:
                    continue
                min_v, max_v = float(col_data.min()), float(col_data.max())
                if min_v == max_v:
                    continue
                sel_range = st.slider(
                    f"{col}", min_value=min_v, max_value=max_v,
                    value=(min_v, max_v),
                )
                df_filtered = df_filtered[
                    df_filtered[col].between(sel_range[0], sel_range[1]) | df_filtered[col].isna()
                ]

if df_filtered.empty:
    st.warning("Tidak ada data yang cocok dengan kombinasi filter saat ini.")
    st.stop()

# =====================================================================
# PART 3B: LIVE PREVIEW UNTUK GRUP YANG BARU DIBUAT
# Begitu ada grup baru dibuat di sidebar ("Kelompokkan Nilai"), chart-nya
# LANGSUNG muncul di sini otomatis - tanpa perlu pindah tab atau klik apapun
# lagi. Ganti pilihan "Nilai" atau "Tampilkan sebagai" di bawah, chart-nya
# otomatis update seketika (ini prinsip kerja Streamlit: setiap widget
# berubah, halaman rerun instan) - jadi terasa "instant" walau bukan
# drag-and-drop beneran.
# =====================================================================
active_grouped_cols = [
    f"{src_col} (Dikelompokkan)"
    for src_col, groups in st.session_state.custom_groups.items()
    if groups and f"{src_col} (Dikelompokkan)" in df_filtered.columns
]

if active_grouped_cols:
    st.markdown("## 🔍 Preview Grup yang Baru Dibuat")
    st.caption(
        "Otomatis muncul begitu kamu bikin grup baru di sidebar. Ganti pengaturan "
        "di bawah, chart langsung update instan."
    )
    for grouped_col in active_grouped_cols:
        with st.container(border=True):
            st.markdown(f"#### {grouped_col}")
            pv1, pv2 = st.columns(2)
            with pv1:
                preview_value_choice = st.selectbox(
                    "Nilai",
                    options=["Jumlah Baris (Count)"] + numeric_cols,
                    key=f"preview_value_{grouped_col}",
                )
            with pv2:
                preview_chart_type = st.selectbox(
                    "Tampilkan sebagai",
                    options=["Bar Chart", "Pie Chart", "Horizontal Bar"],
                    key=f"preview_chart_type_{grouped_col}",
                )

            if preview_value_choice == "Jumlah Baris (Count)":
                preview_data = df_filtered[grouped_col].value_counts().reset_index()
                preview_data.columns = [grouped_col, "Value"]
                preview_label = "Jumlah Baris"
                preview_unit_source = None
            else:
                preview_data = df_filtered.groupby(grouped_col, as_index=False)[preview_value_choice].sum()
                preview_data.columns = [grouped_col, "Value"]
                preview_label = f"Total {preview_value_choice}"
                preview_unit_source = preview_value_choice
            preview_data = preview_data.sort_values("Value", ascending=False)
            preview_data_scaled, preview_unit_caption = scale_for_display(preview_data, "Value", preview_unit_source)

            if preview_chart_type == "Bar Chart":
                fig_preview = px.bar(
                    preview_data_scaled, x=grouped_col, y="Value", color=grouped_col,
                    text="Value", title=f"{preview_label} per {grouped_col}",
                )
                fig_preview.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")
            elif preview_chart_type == "Pie Chart":
                fig_preview = px.pie(
                    preview_data_scaled, names=grouped_col, values="Value", color=grouped_col,
                    title=f"{preview_label} per {grouped_col}",
                )
                fig_preview.update_traces(
                    texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
                    hovertemplate="%{label}<br>Persentase: %{percent:.2%}<br>Nilai: %{value:,.2~f}<extra></extra>",
                )
            else:
                preview_data_h = preview_data_scaled.sort_values("Value")
                fig_preview = px.bar(
                    preview_data_h, x="Value", y=grouped_col, color=grouped_col, orientation="h",
                    text="Value", title=f"{preview_label} per {grouped_col}",
                )
                fig_preview.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")

            show_chart(
                fig_preview, n_categories=len(preview_data), is_pie=(preview_chart_type == "Pie Chart"),
                unit_caption=preview_unit_caption,
            )

    st.markdown("---")


# =====================================================================
# PART 4: KPI SECTION
# =====================================================================

st.markdown("## 📌 Overview")

kpi_cols = st.columns(5)
kpi_cols[0].metric("Total Rows", f"{len(df_filtered):,}")
kpi_cols[1].metric("Total Columns", f"{df_filtered.shape[1]:,}")
kpi_cols[2].metric("Numeric Columns", f"{len(numeric_cols):,}")
kpi_cols[3].metric("Date Columns", f"{len(date_cols):,}")

total_unique_categories = sum(df_filtered[c].nunique(dropna=True) for c in category_cols) \
    if category_cols else 0
kpi_cols[4].metric("Unique Categories", f"{total_unique_categories:,}")

if numeric_cols:
    st.markdown("#### Numeric Summary")
    st.caption("Total & rata-rata di bawah ini dihitung dari SELURUH data yang lolos filter "
               "(bukan pecahan per kategori/company).")
    # Tampilkan semua kolom numeric, dipecah per baris berisi 4 kolom (Total + Avg)
    # supaya kolom seperti Volume tetap muncul totalnya walau bukan di 4 kolom pertama.
    cols_per_row = 4
    for row_start in range(0, len(numeric_cols), cols_per_row):
        row_cols = numeric_cols[row_start:row_start + cols_per_row]
        num_kpi_cols = st.columns(len(row_cols) * 2)
        for i, col in enumerate(row_cols):
            total_val = df_filtered[col].sum()
            avg_val = df_filtered[col].mean()
            unit_suffix = f" ({detect_column_unit(col)})" if detect_column_unit(col) else ""
            num_kpi_cols[i * 2].metric(f"Total {col}{unit_suffix}", fmt_num(total_val))
            num_kpi_cols[i * 2 + 1].metric(f"Avg {col}{unit_suffix}", fmt_num(avg_val))

st.markdown("---")


# =====================================================================
# PART 4B: RINGKASAN / SUMMARY (fleksibel, user pilih sendiri)
# =====================================================================

st.markdown("## 📊 Ringkasan (Summary)")
st.caption(
    "Pilih sendiri kolom angka, cara agregasi, dan mau di-breakdown per apa. "
    "Grand Total di bawah ini selalu dihitung dari seluruh data yang lolos filter."
)

if numeric_cols:
    # Opsi breakdown: kolom kategori biasa (Company, Country, dll) DIGABUNG
    # dengan kolom ID/reference number (Contract Ref.No, Shipment No., dll)
    # yang meski bertipe numeric, secara makna adalah identifier -> tetap
    # perlu bisa dipakai buat breakdown/total per kelompok.
    breakdown_options = category_cols + [c for c in id_like_cols if c not in category_cols]

    s1, s2, s3 = st.columns([2, 1.3, 2])

    with s1:
        summary_value_cols = st.multiselect(
            "Kolom angka yang mau diringkas",
            options=numeric_cols,
            default=numeric_cols[:1],
            key="summary_value_cols",
        )
    with s2:
        agg_label_map = {
            "Total (Sum)": "sum",
            "Rata-rata (Average)": "mean",
            "Jumlah Baris (Count)": "count",
            "Minimum": "min",
            "Maximum": "max",
        }
        agg_label = st.selectbox(
            "Jenis Agregasi", options=list(agg_label_map.keys()), key="summary_agg"
        )
        agg_func = agg_label_map[agg_label]
    with s3:
        breakdown_col = st.selectbox(
            "Breakdown per (opsional)",
            options=["(Tidak di-breakdown / Grand Total saja)"] + breakdown_options,
            key="summary_breakdown",
            help="Bisa pilih kolom kategori (Company, Country) ATAU kolom ID/reference "
                 "seperti Contract Ref.No — cocok kalau 1 nilai kolom itu muncul di banyak baris.",
        )

    if summary_value_cols:
        # ------------- GRAND TOTAL (selalu dari seluruh data terfilter) -------------
        st.markdown("#### Grand Total")
        grand_cols = st.columns(len(summary_value_cols))
        for i, col in enumerate(summary_value_cols):
            grand_val = df_filtered[col].count() if agg_func == "count" \
                else getattr(df_filtered[col], agg_func)()
            grand_cols[i].metric(f"{agg_label} - {col}", fmt_num(grand_val))

        # ------------- BREAKDOWN TABLE (kalau user pilih kolom breakdown) -------------
        if breakdown_col != "(Tidak di-breakdown / Grand Total saja)":
            st.markdown(f"#### Breakdown per {breakdown_col}")
            n_unique_breakdown = df_filtered[breakdown_col].nunique(dropna=True)
            st.caption(f"{n_unique_breakdown:,} nilai unik ditemukan pada kolom ini.")

            summary_table = (
                df_filtered.groupby(breakdown_col)[summary_value_cols]
                .agg(agg_func)
                .reset_index()
            )
            # Tambahkan Jumlah Baris per grup - kalau angkanya lebih banyak dari
            # yang diharapkan (misal 1 Contract Ref.No harusnya cuma 2-3 baris
            # tapi muncul 15 baris), itu petunjuk pertama ada duplikat/data ganda.
            row_counts = df_filtered.groupby(breakdown_col).size().reset_index(name="Jumlah Baris")
            summary_table = summary_table.merge(row_counts, on=breakdown_col)
            summary_table = summary_table.sort_values(summary_value_cols[0], ascending=False)

            st.dataframe(summary_table, use_container_width=True, height=400)

            summary_excel_bytes = to_excel_bytes(summary_table)
            st.download_button(
                label="⬇️ Download Ringkasan Ini (Excel)",
                data=summary_excel_bytes,
                file_name=f"summary_by_{breakdown_col}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_summary",
            )

            # ------------- DRILL-DOWN: cek baris mentah penyusun 1 nilai spesifik -------------
            st.markdown("#### 🔎 Cek Detail Baris Mentah")
            st.caption(
                "Kalau totalnya kelihatan janggal (misal beda jauh dari pivot table kamu), "
                "pilih 1 nilai di bawah ini untuk lihat SEMUA baris mentah yang menyusun "
                "angka tersebut — biasanya ketahuan langsung kalau ada baris duplikat."
            )
            drilldown_options = sorted(df_filtered[breakdown_col].dropna().astype(str).unique().tolist())
            drilldown_value = st.selectbox(
                f"Pilih nilai {breakdown_col} untuk dicek detailnya",
                options=["(Pilih salah satu)"] + drilldown_options,
                key="drilldown_value",
            )
            if drilldown_value != "(Pilih salah satu)":
                drilldown_rows = df_filtered[df_filtered[breakdown_col].astype(str) == drilldown_value]
                st.write(
                    f"Ditemukan **{len(drilldown_rows):,} baris mentah** untuk "
                    f"`{breakdown_col} = {drilldown_value}`."
                )
                for col in summary_value_cols:
                    st.write(
                        f"- Total **{col}** dari {len(drilldown_rows):,} baris ini: "
                        f"**{fmt_num(drilldown_rows[col].sum())}**"
                    )
                st.dataframe(drilldown_rows, use_container_width=True, height=350)
    else:
        st.info("Pilih minimal 1 kolom angka di atas untuk melihat ringkasan.")
else:
    st.info("Tidak ada kolom numeric yang terdeteksi untuk dibuat ringkasan.")

st.markdown("---")


# =====================================================================
# PART 5: VISUALIZATION
# =====================================================================

st.markdown("## 📈 Visualization")

# Tentukan sumbu waktu terbaik yang tersedia (date > year+month > year)
time_axis_col = None
time_axis_type = None
df_viz = df_filtered.copy()

if date_cols:
    time_axis_col = date_cols[0]
    time_axis_type = "date"
elif year_cols and month_cols:
    y_col, m_col = year_cols[0], month_cols[0]
    df_viz["_month_num"] = df_viz[m_col].apply(month_to_number)
    df_viz["_time_period"] = pd.to_datetime(
        df_viz[y_col].astype("Int64").astype(str) + "-" +
        df_viz["_month_num"].fillna(1).astype(int).astype(str) + "-01",
        errors="coerce",
    )
    time_axis_col = "_time_period"
    time_axis_type = "date"
elif year_cols:
    time_axis_col = year_cols[0]
    time_axis_type = "year"

tabs = st.tabs(["⏱️ Time Trend", "🏷️ Category", "📉 Distribution", "🔢 Numeric", "🔗 Relationship", "⚖️ Group Comparison"])

# -----------------------------------------------------------------
# A. TIME TREND
# -----------------------------------------------------------------
with tabs[0]:
    if time_axis_col and numeric_cols:
        value_col = st.selectbox("Pilih Numeric untuk Trend", numeric_cols, key="trend_value")
        agg_df = df_viz.groupby(time_axis_col, as_index=False)[value_col].sum()
        agg_df = agg_df.sort_values(time_axis_col)
        agg_df_scaled, trend_unit_caption = scale_for_display(agg_df, value_col, value_col)

        c1, c2 = st.columns(2)
        with c1:
            fig_line = px.line(
                agg_df_scaled, x=time_axis_col, y=value_col, markers=True,
                title=f"{value_col} over Time",
            )
            show_chart(fig_line, n_categories=len(agg_df), unit_caption=trend_unit_caption)
        with c2:
            fig_area = px.area(
                agg_df_scaled, x=time_axis_col, y=value_col,
                title=f"{value_col} over Time (Area)",
            )
            show_chart(fig_area, n_categories=len(agg_df), unit_caption=trend_unit_caption)
    else:
        st.info("Tidak ditemukan kombinasi kolom waktu (date/year/month) dan numeric yang cukup untuk Time Trend.")

# -----------------------------------------------------------------
# B. CATEGORY ANALYSIS
# -----------------------------------------------------------------
with tabs[1]:
    if category_cols and numeric_cols:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            cat_col = st.selectbox("Pilih Category", category_cols, key="cat_col_bar")
        with cc2:
            val_col = st.selectbox("Pilih Numeric", numeric_cols, key="cat_val_bar")
        with cc3:
            top_n = st.selectbox("Tampilkan Top", options=[1, 3, 5, 10, 15, 20], index=3, key="top_n_bar")

        cat_agg = df_filtered.groupby(cat_col, as_index=False)[val_col].sum()
        cat_agg = cat_agg.sort_values(val_col, ascending=False)
        cat_agg_scaled, cat_unit_caption = scale_for_display(cat_agg, val_col, val_col)

        b1, b2 = st.columns(2)
        with b1:
            fig_bar = px.bar(
                cat_agg_scaled, x=cat_col, y=val_col,
                title=f"{cat_col} vs Total {val_col}",
            )
            show_chart(fig_bar, n_categories=len(cat_agg), unit_caption=cat_unit_caption)
        with b2:
            top_n_data = cat_agg.head(top_n).sort_values(val_col)
            top_n_data_scaled, top_n_unit_caption = scale_for_display(top_n_data, val_col, val_col)
            fig_hbar = px.bar(
                top_n_data_scaled, x=val_col, y=cat_col, orientation="h",
                title=f"Top {top_n} {cat_col} by {val_col}",
                text=val_col,
            )
            fig_hbar.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")
            show_chart(fig_hbar, n_categories=len(top_n_data), unit_caption=top_n_unit_caption)
    else:
        st.info("Tidak ditemukan kombinasi kolom category dan numeric untuk Category Analysis.")

# -----------------------------------------------------------------
# C. DISTRIBUTION
# -----------------------------------------------------------------
with tabs[2]:
    if category_cols:
        dc1, dc2 = st.columns(2)
        with dc1:
            dist_col = st.selectbox("Pilih Category untuk Distribusi", category_cols, key="dist_col")
        with dc2:
            dist_value_options = ["Jumlah Baris (Count)"] + numeric_cols
            dist_value_choice = st.selectbox(
                "Berdasarkan", options=dist_value_options, key="dist_value_choice"
            )

        if dist_value_choice == "Jumlah Baris (Count)":
            dist_data = df_filtered[dist_col].value_counts().reset_index()
            dist_data.columns = [dist_col, "Value"]
            value_label = "Jumlah Baris"
            dist_unit_source = None
        else:
            dist_data = df_filtered.groupby(dist_col, as_index=False)[dist_value_choice].sum()
            dist_data.columns = [dist_col, "Value"]
            value_label = f"Total {dist_value_choice}"
            dist_unit_source = dist_value_choice

        dist_data = dist_data.sort_values("Value", ascending=False)

        others_selected_dist = st.multiselect(
            "Gabungkan kategori berikut jadi 'Others' (opsional)",
            options=dist_data[dist_col].astype(str).tolist(),
            key="others_selected_dist",
            help="Pilih kategori mana saja yang mau digabung jadi satu slice 'Others' di pie kanan. "
                 "Pie kiri tetap tampil full tanpa 'Others'.",
        )
        render_full_vs_others_pies(
            dist_data, dist_col, "Value", f"Distribution of {dist_col} ({value_label})", others_selected_dist,
            unit_source_col=dist_unit_source,
        )

        with st.expander("📋 Lihat angka detail per kategori"):
            dist_data_display = dist_data.copy()
            total_val = dist_data_display["Value"].sum()
            dist_data_display["Persentase"] = (dist_data_display["Value"] / total_val * 100).map(lambda x: f"{x:.2f}%")
            dist_data_display["Value"] = dist_data_display["Value"].map(fmt_num)
            dist_data_display.columns = [dist_col, value_label, "Persentase"]
            st.dataframe(dist_data_display, use_container_width=True)

        # -----------------------------------------------------------------
        # DRILL-DOWN: kalau kolom yang dipilih (dist_col) punya nilai hasil
        # "Kelompokkan Nilai -> digabung ke kolom ini" (misal kolom 'Kota'
        # kedapatan nilai 'Jakarta' yang sebenarnya gabungan dari 'abc',
        # 'abcd', 'abcde'), tampilkan pie chart rincian kedua yang membedah
        # isi asli grup tsb - persis seperti pie 'Jakarta' pecah jadi
        # abc/abcd/abcde di sebelah pie 'Kota' pada sketsa.
        # -----------------------------------------------------------------
        merge_groups_for_dist = get_merge_groups_for_target(dist_col)
        if merge_groups_for_dist:
            st.markdown("---")
            st.markdown("#### 🔍 Rincian Grup (Drill-Down)")
            st.caption(
                f"Kolom **{dist_col}** punya nilai hasil gabungan dari fitur 'Kelompokkan Nilai'. "
                "Pilih salah satu grup di bawah untuk lihat rincian isi aslinya "
                "(contoh: 'Jakarta' dirinci lagi jadi abc, abcd, abcde)."
            )
            drill_group_names = [g["name"] for g in merge_groups_for_dist]
            selected_drill_group = st.selectbox(
                "Pilih grup untuk dirinci", options=drill_group_names, key="drill_group_select"
            )
            drill_info = next(g for g in merge_groups_for_dist if g["name"] == selected_drill_group)
            drill_src_col = drill_info["src_col"]

            df_drill = df_filtered[df_filtered[dist_col].astype(str) == selected_drill_group]

            if drill_src_col in df_drill.columns and not df_drill.empty:
                if dist_value_choice == "Jumlah Baris (Count)":
                    drill_data = df_drill[drill_src_col].value_counts().reset_index()
                    drill_data.columns = [drill_src_col, "Value"]
                else:
                    drill_data = df_drill.groupby(drill_src_col, as_index=False)[dist_value_choice].sum()
                    drill_data.columns = [drill_src_col, "Value"]
                drill_data = drill_data.sort_values("Value", ascending=False)
                others_selected_drill = st.multiselect(
                    "Gabungkan kategori berikut jadi 'Others' (opsional)",
                    options=drill_data[drill_src_col].astype(str).tolist(),
                    key="others_selected_drill",
                    help="Pilih kategori mana saja yang mau digabung jadi satu slice 'Others' di pie kanan.",
                )
                render_full_vs_others_pies(
                    drill_data, drill_src_col, "Value",
                    f"Rincian '{selected_drill_group}' per {drill_src_col} ({value_label})",
                    others_selected_drill,
                    unit_source_col=dist_unit_source,
                )

                with st.expander(f"📋 Lihat angka detail rincian '{selected_drill_group}'"):
                    drill_data_display = drill_data.copy()
                    total_drill_val = drill_data_display["Value"].sum()
                    drill_data_display["Persentase"] = (
                        drill_data_display["Value"] / total_drill_val * 100
                    ).map(lambda x: f"{x:.2f}%")
                    drill_data_display["Value"] = drill_data_display["Value"].map(fmt_num)
                    drill_data_display.columns = [drill_src_col, value_label, "Persentase"]
                    st.dataframe(drill_data_display, use_container_width=True)
            else:
                st.info(f"Tidak ada data untuk grup '{selected_drill_group}' pada filter/kolom saat ini.")
    else:
        st.info("Tidak ditemukan kolom category untuk chart distribusi.")

# -----------------------------------------------------------------
# D. NUMERIC ANALYSIS
# -----------------------------------------------------------------
with tabs[3]:
    if numeric_cols:
        num_col = st.selectbox("Pilih Numeric Column", numeric_cols, key="num_analysis_col")
        show_unit_note(num_col)
        n1, n2 = st.columns(2)
        with n1:
            fig_hist = px.histogram(df_filtered, x=num_col, title=f"Histogram of {num_col}")
            show_chart(fig_hist)
        with n2:
            fig_box = px.box(df_filtered, y=num_col, title=f"Box Plot of {num_col}")
            show_chart(fig_box)
    else:
        st.info("Tidak ditemukan kolom numeric untuk analisis distribusi numerik.")

# -----------------------------------------------------------------
# E. RELATIONSHIP
# -----------------------------------------------------------------
with tabs[4]:
    if len(numeric_cols) >= 2:
        r1, r2 = st.columns(2)
        with r1:
            x_col = st.selectbox("Sumbu X", numeric_cols, index=0, key="scatter_x")
        with r2:
            y_default_idx = 1 if len(numeric_cols) > 1 else 0
            y_col = st.selectbox("Sumbu Y", numeric_cols, index=y_default_idx, key="scatter_y")

        color_arg = category_cols[0] if category_cols else None
        fig_scatter = px.scatter(
            df_filtered, x=x_col, y=y_col, color=color_arg,
            title=f"{x_col} vs {y_col}",
        )
        show_chart(fig_scatter)
    else:
        st.info("Minimal dibutuhkan 2 kolom numeric untuk membuat Scatter Plot.")

# -----------------------------------------------------------------
# F. GROUP COMPARISON (generic - berlaku untuk kolom & nilai apapun,
# tidak terikat ke istilah/kasus bisnis tertentu)
# -----------------------------------------------------------------
with tabs[5]:
    st.caption(
        "Bagi data jadi 2 kelompok berdasarkan kolom & nilai pilihan kamu sendiri "
        "(misal Domestik vs Ekspor, Aktif vs Nonaktif, atau apapun sesuai isi file), "
        "lalu breakdown salah satu kelompoknya lebih detail per sub-kategori dengan warna berbeda."
    )
    if category_cols and numeric_cols:
        gc1, gc2 = st.columns(2)
        with gc1:
            group_class_col = st.selectbox(
                "Kolom Klasifikasi (untuk membagi 2 kelompok)",
                options=category_cols,
                key="group_class_col",
                help="Kolom kategori yang mau dipakai buat misahin data jadi 2 kelompok, "
                     "misal kolom Region, Tipe, Status, dll.",
            )
        with gc2:
            group_value_col = st.selectbox(
                "Kolom Angka yang Dianalisis",
                options=numeric_cols,
                key="group_value_col",
            )
        show_unit_note(group_value_col)

        group_class_options = sorted(df_filtered[group_class_col].dropna().astype(str).unique().tolist())
        group_a_name = st.text_input(
            "Nama untuk Kelompok A (bebas, cuma label tampilan)",
            value="Kelompok A", key="group_a_label",
        )
        selected_group_a_values = st.multiselect(
            f"Pilih nilai yang masuk '{group_a_name}' (sisanya otomatis jadi kelompok satunya)",
            options=group_class_options,
            key="group_a_selected_values",
        )

        other_category_cols = [c for c in category_cols if c != group_class_col]
        group_sub_col = None
        if other_category_cols:
            group_sub_col = st.selectbox(
                "Kolom Sub-Kategori (opsional, buat breakdown lebih detail)",
                options=["(Tidak dipecah lagi)"] + other_category_cols,
                key="group_sub_col",
            )
            if group_sub_col == "(Tidak dipecah lagi)":
                group_sub_col = None

        if selected_group_a_values:
            df_group_a = df_filtered[df_filtered[group_class_col].astype(str).isin(selected_group_a_values)]
            df_group_b = df_filtered[~df_filtered[group_class_col].astype(str).isin(selected_group_a_values)]
            group_b_name = f"Selain {group_a_name}"

            st.markdown(f"#### Perbandingan {group_a_name} vs {group_b_name}")
            comp1, comp2 = st.columns(2)
            group_a_total = df_group_a[group_value_col].sum()
            group_b_total = df_group_b[group_value_col].sum()
            grand_total_group = group_a_total + group_b_total
            comp1.metric(
                f"Total {group_value_col} - {group_a_name}",
                fmt_num(group_a_total),
                f"{(group_a_total/grand_total_group*100 if grand_total_group else 0):.2f}% dari total",
            )
            comp2.metric(
                f"Total {group_value_col} - {group_b_name}",
                fmt_num(group_b_total),
                f"{(group_b_total/grand_total_group*100 if grand_total_group else 0):.2f}% dari total",
            )

            if group_sub_col:
                st.markdown(f"#### Breakdown {group_a_name} per {group_sub_col}")
                group_breakdown = (
                    df_group_a.groupby(group_sub_col, as_index=False)[group_value_col]
                    .sum()
                    .sort_values(group_value_col, ascending=False)
                )
                group_row_counts = df_group_a.groupby(group_sub_col).size().reset_index(name="Jumlah Baris")
                group_breakdown = group_breakdown.merge(group_row_counts, on=group_sub_col)

                others_selected_comparison = st.multiselect(
                    "Gabungkan kategori berikut jadi 'Others' di pie chart (opsional)",
                    options=group_breakdown[group_sub_col].astype(str).tolist(),
                    key="others_selected_comparison",
                    help="Pilih kategori mana saja yang mau digabung jadi satu slice 'Others'.",
                )

                render_full_vs_others_pies(
                    group_breakdown, group_sub_col, group_value_col,
                    f"Distribusi {group_a_name} per {group_sub_col}",
                    others_selected_comparison,
                    unit_source_col=group_value_col,
                )

                group_breakdown_scaled, group_bar_unit_caption = scale_for_display(
                    group_breakdown, group_value_col, group_value_col
                )
                fig_group_bar = px.bar(
                    group_breakdown_scaled, x=group_sub_col, y=group_value_col,
                    color=group_sub_col,
                    text=group_value_col,
                    title=f"Total {group_value_col} per {group_sub_col}",
                )
                fig_group_bar.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")
                show_chart(fig_group_bar, n_categories=len(group_breakdown), unit_caption=group_bar_unit_caption)

                st.markdown("##### 📋 Data Detail")
                group_breakdown_display = group_breakdown.copy()
                total_group_val = group_breakdown_display[group_value_col].sum()
                group_breakdown_display["Persentase"] = (
                    group_breakdown_display[group_value_col] / total_group_val * 100
                ).map(lambda x: f"{x:.2f}%")
                group_breakdown_display[group_value_col] = group_breakdown_display[group_value_col].map(fmt_num)
                st.dataframe(group_breakdown_display, use_container_width=True)
            else:
                st.info("Pilih kolom Sub-Kategori di atas untuk lihat breakdown warna-warni per grup.")
        else:
            st.info(f"Pilih minimal 1 nilai yang masuk '{group_a_name}' di atas.")
    else:
        st.info("Butuh minimal 1 kolom category dan 1 kolom numeric untuk Group Comparison.")

st.markdown("---")


# =====================================================================
# PART 6: DATA PREVIEW & EXPORT
# =====================================================================

st.markdown("## 🧾 Data Preview")
st.caption(f"Menampilkan {len(df_filtered):,} baris setelah filter (dari total {len(df_raw):,} baris).")
st.dataframe(df_filtered, use_container_width=True, height=380)


excel_bytes = to_excel_bytes(df_filtered)

st.download_button(
    label="⬇️ Download Filtered Excel",
    data=excel_bytes,
    file_name="filtered_dashboard_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")
st.caption(
    "Dashboard ini bersifat generik — struktur filter, KPI, dan chart menyesuaikan "
    "otomatis terhadap kolom apa pun yang terdeteksi dari file Excel yang diupload."
)