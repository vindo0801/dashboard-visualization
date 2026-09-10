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
import datetime

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

# Nama bulan Bahasa Indonesia juga dikenali (banyak file Excel Indonesia
# pakai 'Januari', 'Februari', dst, bukan nama bulan Inggris) - ini yang
# bikin kolom Month kedetect tapi filternya kosong/ga muncul sebelumnya.
MONTH_ALIAS_ID = [
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
]
MONTH_ALIAS.update({m: i + 1 for i, m in enumerate(MONTH_ALIAS_ID)})
MONTH_ALIAS.update({m[:3]: i + 1 for i, m in enumerate(MONTH_ALIAS_ID)})
# Singkatan umum yang beda dari 3-huruf standar
MONTH_ALIAS.update({"agt": 8, "ags": 8, "des": 12, "nov": 11, "okt": 10})

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


# Jumlah minimum kolom ber-nama-tanggal supaya dianggap "wide time-series"
# beneran (bukan cuma kebetulan ada 1-2 kolom bernama tanggal).
WIDE_PERIOD_MIN_COLS = 6


def detect_wide_period_columns(df: pd.DataFrame):
    """Deteksi kolom-kolom yang NAMA-nya sendiri berupa tanggal/periode
    (misal header kolom '2000-01-01', '2000-02-01', dst - satu kolom per
    bulan/periode/tahun). Ini format WIDE yang umum dipakai laporan trade/
    statistik (misal Global Trade Tracker, BPS, data ekspor-impor): tiap
    baris = 1 kombinasi kategori, tiap kolom = 1 titik waktu, isinya angka.
    Mendukung 2 granularitas:
    - 'date': header kolom berupa tanggal lengkap (misal sheet bulanan)
    - 'year': header kolom berupa angka tahun murni (misal sheet tahunan,
      contoh header '2000', '2001', dst - TANPA info bulan)
    Kalau jumlah kolom seperti ini banyak (>= WIDE_PERIOD_MIN_COLS),
    dashboard akan otomatis 'unpivot' (melt) supaya bisa dianalisis
    dengan fitur filter tanggal/tahun/tren yang sama seperti data format
    panjang biasa - generic, berlaku untuk sheet Excel apapun dengan pola
    ini, bukan cuma untuk 1 file tertentu.
    Return (granularity, period_cols) - granularity None kalau ga kedetect."""
    date_period_cols = []
    year_period_cols = []
    for col in df.columns:
        if isinstance(col, (pd.Timestamp, datetime.date, datetime.datetime)):
            date_period_cols.append(col)
            continue
        col_str = str(col).strip()
        # Cocok kalau nama kolom BENAR-BENAR berpola tanggal (bukan sekadar
        # angka biasa) - misal '2000-01-01', '2000-01', 'Jan-2000', 'Jan 2000'.
        # Toleran terhadap komponen jam (misal '2000-01-01 00:00:00' atau
        # '2000-01-01T00:00:00') supaya kolom header bertipe datetime yang
        # SUDAH terlanjur berubah jadi teks (misal lewat str()/to_string())
        # tetap kedetect dengan benar, bukan cuma yang masih objek datetime asli.
        looks_like_date = bool(
            re.match(r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$", col_str)
            or re.match(r"^[A-Za-z]{3,9}[\s\-]\d{2,4}$", col_str)
        )
        if looks_like_date:
            parsed = pd.to_datetime(col_str, errors="coerce")
            if pd.notna(parsed):
                date_period_cols.append(col)
                continue
        # Cek pola TAHUN MURNI (misal header berupa angka 2000, 2000.0,
        # atau string '2000') - dipakai kalau sheet-nya wide-format tahunan.
        year_val = None
        if isinstance(col, (int, float, np.integer, np.floating)) and not isinstance(col, bool):
            if float(col).is_integer():
                year_val = int(col)
        elif re.fullmatch(r"(19|20)\d{2}", col_str):
            year_val = int(col_str)
        elif re.fullmatch(r"(19|20)\d{2}\.0", col_str):
            # Header tahun bertipe float yang sudah kadung di-str()-kan,
            # misal '2000.0' (dari kolom Excel bertipe float 2000.0).
            year_val = int(float(col_str))
        if year_val is not None and 1900 <= year_val <= 2100:
            year_period_cols.append(col)

    if len(date_period_cols) >= WIDE_PERIOD_MIN_COLS:
        return "date", date_period_cols
    if len(year_period_cols) >= WIDE_PERIOD_MIN_COLS:
        return "year", year_period_cols
    return None, []


def detect_sheet_unit_hint(file_bytes: bytes, sheet_name, header_row: int) -> str:
    """Coba cari keterangan satuan (misal '(Tonnes)') dari baris-baris DI
    ATAS baris header - banyak laporan trade/statistik nulis satuan di
    situ, terpisah dari nama kolom. Return '' kalau ga ketemu apapun."""
    if header_row <= 0:
        return ""
    try:
        preview = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None, nrows=header_row)
    except Exception:
        return ""
    generic_words = {"source", "note", "notes", "data availability", "contents"}
    for _, row in preview.iterrows():
        for cell in row:
            if isinstance(cell, str):
                m = re.fullmatch(r"\(?\s*([A-Za-z][A-Za-z /]{1,25}?)\s*\)?", cell.strip())
                if m:
                    candidate = m.group(1).strip()
                    if candidate.lower() not in generic_words and len(candidate) >= 2:
                        return candidate
    return ""


def melt_wide_period_columns(
    df: pd.DataFrame, period_cols: list, period_col_name: str, value_col_name: str, granularity: str = "date"
) -> pd.DataFrame:
    """Ubah kolom-kolom periode (format WIDE, 1 kolom = 1 titik waktu)
    jadi format PANJANG (long format): 1 baris = 1 kombinasi
    id + periode + nilai. Kolom id lain (misal 'Import country', 'First',
    'Last') dipertahankan apa adanya. Setelah di-melt, dashboard bisa
    langsung pakai fitur filter tanggal/bulan/tahun dan visualisasi tren
    seperti data format panjang biasa.
    `granularity`: 'date' (kolom hasil jadi datetime lengkap) atau
    'year' (kolom hasil jadi angka tahun murni, buat sheet tahunan)."""
    id_vars = [c for c in df.columns if c not in period_cols]
    melted = df.melt(id_vars=id_vars, value_vars=period_cols, var_name=period_col_name, value_name=value_col_name)
    if granularity == "date":
        melted[period_col_name] = pd.to_datetime(melted[period_col_name], errors="coerce")
    else:
        melted[period_col_name] = pd.to_numeric(melted[period_col_name], errors="coerce")
        melted = melted.dropna(subset=[period_col_name])
        melted[period_col_name] = melted[period_col_name].astype(int)
    return melted


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
                # PENTING: pandas.to_datetime salah nge-parse nama bulan
                # polos (misal 'Jan', 'Feb', 'December') jadi tanggal
                # dengan tahun DEFAULT 0001 (karena ga ada info tahun
                # beneran) - misal 'Jan' -> 0001-01-01. Kalau semua hasil
                # parse-nya punya tahun 1 yang sama, ini BUKAN kolom
                # tanggal beneran (kemungkinan besar kolom bulan/Month
                # yang ke-detect ganda), jadi di-skip dari date_cols.
                valid_years = converted.dropna().dt.year
                if not valid_years.empty and valid_years.nunique() == 1 and valid_years.iloc[0] == 1:
                    continue
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
    """Konversi nilai bulan (nama, angka, ATAU tanggal penuh) menjadi
    angka 1-12. Kalau kolom 'Month' ternyata isinya tanggal lengkap
    (misal '2024-01-01' atau objek Timestamp/date), otomatis diambil
    bagian bulannya saja - jadi filter tetap tampil 'January' dkk,
    bukan tanggal mentah 'yyyy-mm-dd'."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        v = int(val)
        return v if 1 <= v <= 12 else np.nan
    if isinstance(val, (pd.Timestamp, datetime.date, datetime.datetime)):
        return val.month
    s = str(val).strip().lower()
    if s in MONTH_ALIAS:
        return MONTH_ALIAS[s]
    # Kalau isinya angka dalam bentuk teks, misal "5" atau "05"
    if s.isdigit():
        v = int(s)
        return v if 1 <= v <= 12 else np.nan
    # Kalau isinya string tanggal lengkap (misal "2024-01-01", "01/02/2024",
    # "Jan-2024") - coba parse jadi tanggal, lalu ambil bulannya.
    parsed = pd.to_datetime(val, errors="coerce")
    if pd.notna(parsed):
        return parsed.month
    return np.nan


def detect_category_columns(df: pd.DataFrame, exclude: list) -> list:
    """Deteksi kolom kategorikal: object/string dengan unique value TERBATAS
    (<= CATEGORY_MAX_UNIQUE). Dipakai KHUSUS untuk sidebar Category Filter
    (checkbox), supaya sidebar nggak kebanjiran ratusan/ribuan checkbox
    kalau ada kolom dengan unique value yang sangat banyak (misal nama
    kapal, nama customer, dst)."""
    cat_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if is_text_dtype(df[col]) or str(df[col].dtype).startswith("category"):
            n_unique = df[col].nunique(dropna=True)
            if 0 < n_unique <= CATEGORY_MAX_UNIQUE:
                cat_cols.append(col)
    return cat_cols


def detect_all_text_columns(df: pd.DataFrame, exclude: list) -> list:
    """Semua kolom teks TANPA batas jumlah unique value - dipakai untuk
    dropdown single-select di mana-mana (Category Analysis, Distribution,
    Group Comparison, Kelompokkan Nilai, Summary breakdown, dll) yang aman
    nampung ribuan pilihan sekalipun karena cuma dropdown biasa, bukan
    checkbox. BEDA dari detect_category_columns yang sengaja dibatasi
    CATEGORY_MAX_UNIQUE khusus buat sidebar Category Filter."""
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if is_text_dtype(df[col]) or str(df[col].dtype).startswith("category"):
            if df[col].nunique(dropna=True) > 0:
                cols.append(col)
    return cols


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


# Kata skala yang dikenali (Indonesia & Inggris) beserta pengalinya - termasuk
# singkatan umum yang biasa dipakai di laporan komoditas/trade (batu bara,
# minyak, dll), misal 'mn t' (million ton), 'bn t' (billion ton).
SCALE_WORDS = {
    "thousand": 1_000, "ribu": 1_000, "th": 1_000, "k": 1_000,
    "million": 1_000_000, "juta": 1_000_000, "mn": 1_000_000, "mio": 1_000_000,
    "billion": 1_000_000_000, "miliar": 1_000_000_000, "milyar": 1_000_000_000,
    "bn": 1_000_000_000, "bio": 1_000_000_000,
}
# Diurutkan dari yang paling panjang, dipakai buat cek token GABUNGAN tanpa
# spasi (misal 'mnt' = 'mn' + 't', 'bnusd' = 'bn' + 'usd'). SENGAJA cuma
# singkatan yang cukup panjang & ga ambigu ('mn','bn','mio','bio') - huruf
# tunggal kayak 'k'/'th' DIKECUALIKAN dari mode gabungan ini karena rawan
# ketuker sama awalan satuan umum yang sudah lazim (misal 'kg' kilogram,
# 'km' kilometer, 'kt' kiloton) - itu tetap dibaca sebagai satuan utuh,
# bukan dipecah jadi 'ribu + g/m/t'.
_SAFE_COMBINED_PREFIXES = ["million", "billion", "thousand", "mio", "bio", "mn", "bn"]
_SCALE_PREFIXES_BY_LEN = sorted(_SAFE_COMBINED_PREFIXES, key=len, reverse=True)


def _split_scale_prefix(token: str):
    """Coba pisahkan token gabungan macam 'mnt' -> ('mn', 't')."""
    tl = token.lower()
    for prefix in _SCALE_PREFIXES_BY_LEN:
        if tl.startswith(prefix) and len(tl) > len(prefix):
            rest = token[len(prefix):]
            if rest.isalpha():
                return prefix, rest
    return None, None


def parse_unit_parts(unit_text: str):
    """Pecah keterangan satuan jadi (skala_eksplisit, kata_skala, satuan_dasar).
    Misal: 'Million MT' -> (1_000_000, 'Million', 'MT')
           'MT'         -> (None, None, 'MT')
           'USD Thousand' -> (1_000, 'Thousand', 'USD')
           'mn t'       -> (1_000_000, 'mn', 't')
           'mnt'        -> (1_000_000, 'mn', 't')   (token gabungan tanpa spasi)
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
        # Coba token gabungan tanpa spasi, misal 'mnt' -> scale 'mn' + sisa 't'
        prefix, rest_of_token = _split_scale_prefix(tok)
        if prefix:
            rest_tokens = tokens[:i] + ([rest_of_token] if rest_of_token else []) + tokens[i + 1:]
            rest = " ".join(rest_tokens).strip()
            return SCALE_WORDS[prefix], prefix, rest
    return None, None, unit_text


def auto_scale_factor(max_abs_value: float):
    """Auto-detect pembagi (Thousand/Million/Billion) berdasarkan besarnya
    angka, supaya angka yang ditampilkan di chart lebih ringkas & gampang
    dibaca (misal 22.000.123 jadi 22.000 dengan keterangan 'In Th').
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


# Bentuk singkat buat caption satuan di chart - lebih presentable & lazim
# dipakai di laporan komoditas/trade (misal 'Mn T' = Million Ton), daripada
# nulis lengkap 'Million'/'Billion'/'Thousand'.
SCALE_ABBREV = {1_000: "Th", 1_000_000: "Mn", 1_000_000_000: "Bn"}
# Huruf depan yang dianggap "redundan" kalau base unit-nya diawali huruf
# yang sama dengan skala yang UDAH kesebut duluan di caption - misal skala
# 'Million' + unit 'MT' -> 'M' di depan 'MT' jadi dobel nyebut Million,
# jadi dibuang sisain 'T' aja. Cuma berlaku utk Million/Billion - BUKAN
# Thousand, karena 'KT' (kiloton) itu satuan valid tersendiri, bukan
# redundan, jadi ga boleh ikut dipotong.
_REDUNDANT_SCALE_LETTER = {1_000_000: "m", 1_000_000_000: "b"}


def _strip_redundant_scale_letter(base_unit: str, scale_val: int) -> str:
    """Buang huruf depan base_unit kalau itu dobel nyebut skala yang udah
    ada di caption (misal 'MT' + skala Million -> 'T')."""
    letter = _REDUNDANT_SCALE_LETTER.get(scale_val)
    if not letter or not base_unit or len(base_unit) <= 1:
        return base_unit
    if base_unit[0].lower() == letter:
        return base_unit[1:]
    return base_unit


def get_display_scale(col_name: str, series: pd.Series):
    """Tentukan (divisor, caption) buat nampilin angka kolom `col_name`
    dengan data `series` di chart:
    - Kalau nama kolom SUDAH eksplisit nyebut skala (misal 'Volume (Million MT)'
      atau 'Volume (mn t)'), dianggap datanya SUDAH dalam skala itu -> ga
      dibagi lagi (divisor=1), caption disingkat & dibersihkan jadi
      'In Mn T' (bukan 'In Million MT' yang lebih panjang).
    - Kalau BELUM ada skala eksplisit (misal cuma 'Volume (MT)' atau 'Volume'
      dengan angka mentah besar seperti 22.000.123), auto-detect skala yang
      pas dari besarnya angka, angkanya DIBAGI biar ringkas, caption jadi
      misal 'In Th MT' - inilah yang bikin '22.000.123' -> '22.000'."""
    unit_text = detect_column_unit(col_name)
    explicit_scale_val, explicit_scale_word, base_unit = parse_unit_parts(unit_text)

    if explicit_scale_val:
        abbrev = SCALE_ABBREV.get(explicit_scale_val, explicit_scale_word)
        trimmed_unit = _strip_redundant_scale_letter(base_unit, explicit_scale_val)
        caption = f"In {abbrev} {trimmed_unit}".strip()
        return 1, caption

    max_abs = series.abs().max() if series is not None and len(series) else None
    auto_div, auto_label = auto_scale_factor(max_abs)
    if auto_div == 1:
        return 1, (f"In {base_unit}" if base_unit else "")

    abbrev = SCALE_ABBREV.get(auto_div, auto_label)
    trimmed_unit = _strip_redundant_scale_letter(base_unit, auto_div)
    caption = f"In {abbrev}" + (f" {trimmed_unit}" if trimmed_unit else "")
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
    `unit_caption`: kalau diisi (misal 'In Mn T'), digabung jadi SUBTITLE
    rapi di bawah judul chart (bukan anotasi terpisah yang bisa numpuk)."""
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
        # Subtitle resmi di bawah judul (bukan anotasi terpisah) - supaya
        # selalu rapi & ga akan tabrakan sama judul chart apapun panjangnya.
        current_title = ""
        if fig.layout.title and fig.layout.title.text:
            current_title = fig.layout.title.text
        height += 26
        fig.update_layout(
            title=dict(
                text=f"{current_title}<br><span style='font-size:13px;color:#666666'>{unit_caption}</span>",
            ),
            height=height,
            margin=dict(t=(fig.layout.margin.t or 70) + 26),
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


PIE_DEFAULT_PALETTE = px.colors.qualitative.Plotly


def pick_pie_colors_and_pull(categories: list, section_key: str):
    """Tampilkan kontrol interaktif (dalam expander) untuk kustomisasi pie
    chart: warna sendiri per kategori, pilih slice mana yang mau "ditarik
    keluar" (explode/pull), dan rotasi awal chart. `section_key` harus unik
    per pemanggilan (misal nama tab + nama kolom) supaya widget & warna yang
    diingat tidak bentrok/ketimpa dengan pie chart lain di tab lain.
    Return: (color_map: dict, pull_categories: list, pull_amount: float, rotation_deg: int)
    """
    categories = [str(c) for c in categories]
    color_map = {}
    with st.expander(f"🎨 Kustomisasi Tampilan Pie Chart ({section_key})"):
        st.caption(
            "Pilih warna sendiri per kategori, tarik keluar (explode) slice tertentu "
            "buat highlight, dan atur rotasi awal chart kalau perlu."
        )
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.markdown("**Warna per kategori**")
            color_grid = st.columns(3)
            for i, cat in enumerate(categories):
                default_color = PIE_DEFAULT_PALETTE[i % len(PIE_DEFAULT_PALETTE)]
                widget_key = f"color_{section_key}_{cat}"
                chosen = color_grid[i % 3].color_picker(cat, value=default_color, key=widget_key)
                color_map[cat] = chosen
        with cc2:
            st.markdown("**Explode & Rotasi**")
            pull_categories = st.multiselect(
                "Tarik keluar (explode) slice ini",
                options=categories,
                key=f"pull_cats_{section_key}",
                help="Slice yang dipilih akan sedikit 'ditarik keluar' dari lingkaran pie untuk menonjolkan fokus.",
            )
            pull_amount = 0.0
            if pull_categories:
                pull_amount = st.slider(
                    "Besar tarikan slice", min_value=0.02, max_value=0.4, value=0.12, step=0.02,
                    key=f"pull_amt_{section_key}",
                )
            rotation_deg = st.slider(
                "Rotasi awal chart (derajat)", min_value=0, max_value=360, value=0, step=15,
                key=f"rotation_{section_key}",
                help="Mengubah posisi mulai slice pertama (jam 12 = 0 derajat). Murni estetika, tidak mengubah data.",
            )
    return color_map, pull_categories, pull_amount, rotation_deg


def apply_pie_customization(fig, data: pd.DataFrame, label_col: str, color_map: dict = None,
                             pull_categories: list = None, pull_amount: float = 0.0,
                             rotation_deg: int = 0) -> None:
    """Terapkan pull (explode) & rotation ke trace pie yang sudah dibuat px.pie.
    Warna sudah diterapkan lewat color_discrete_map saat px.pie dipanggil, jadi
    fungsi ini fokus ke pull & rotation yang butuh urutan data asli."""
    pull_categories = pull_categories or []
    if pull_categories and pull_amount:
        pull_array = [pull_amount if str(v) in pull_categories else 0 for v in data[label_col]]
        fig.update_traces(pull=pull_array)
    if rotation_deg:
        fig.update_traces(rotation=rotation_deg)


def render_full_vs_others_pies(
    data: pd.DataFrame, label_col: str, value_col: str, title_prefix: str, others_selected: list,
    unit_source_col: str = None, color_map: dict = None, pull_categories: list = None,
    pull_amount: float = 0.0, rotation_deg: int = 0,
) -> None:
    """Tampilkan 2 pie chart perbandingan: kiri/atas versi FULL (semua
    kategori apa adanya), kanan/bawah versi dengan kategori terpilih
    digabung jadi 'Others'. Kalau kategorinya banyak, ditumpuk VERTIKAL
    (full width) bukan berdampingan, supaya label outside-nya ada ruang
    lebih lega dan ga kepotong.
    `unit_source_col`: nama kolom numeric ASLI (buat deteksi satuan &
    auto-scale angka gede, misal 22.000.123 -> 22.000 + caption
    'In Th MT'). Isi None kalau value_col bukan kolom angka asli
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
        fig_full = px.pie(
            data, names=label_col, values=value_col, title=f"{title_prefix} - Full",
            color=label_col, color_discrete_map=color_map,
        )
        fig_full.update_traces(
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
            hovertemplate="%{label}<br>Persentase: %{percent:.2%}<br>Nilai: %{value:,.2~f}<extra></extra>",
        )
        apply_pie_customization(fig_full, data, label_col, color_map, pull_categories, pull_amount, rotation_deg)
        show_chart(fig_full, n_categories=len(data), is_pie=True, unit_caption=unit_caption)
    with pc_others:
        st.caption("🗂️ Dengan 'Others'")
        data_with_others = group_selected_as_others(data, label_col, value_col, others_selected)
        fig_others = px.pie(
            data_with_others, names=label_col, values=value_col, title=f"{title_prefix} - Others",
            color=label_col, color_discrete_map=color_map,
        )
        fig_others.update_traces(
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
            hovertemplate="%{label}<br>Persentase: %{percent:.2%}<br>Nilai: %{value:,.2~f}<extra></extra>",
        )
        apply_pie_customization(
            fig_others, data_with_others, label_col, color_map, pull_categories, pull_amount, rotation_deg
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

    # -----------------------------------------------------------------
    # WIDE TIME-SERIES AUTO-UNPIVOT: kalau sheet-nya ternyata format WIDE
    # (1 kolom = 1 bulan/periode, header-nya literal tanggal/tahun - umum
    # di laporan trade/statistik seperti Global Trade Tracker/BPS), otomatis
    # di-'melt' jadi format panjang (1 baris = 1 titik waktu) supaya bisa
    # dipakai penuh oleh seluruh fitur dashboard (filter tanggal/bulan/
    # tahun, KPI, chart). Generic - berlaku untuk sheet Excel manapun
    # dengan pola wide time-series, bukan cuma 1 file tertentu.
    #
    # PENTING: ini HARUS dijalankan SEBELUM clean_column_names(), karena
    # kalau header kolom aslinya berupa objek datetime asli (umum kalau
    # Excel-nya nyimpen tanggal sebagai date cell, bukan teks), begitu
    # clean_column_names() menjalankan str() ke nama kolom, objek datetime
    # itu berubah jadi teks lengkap dengan jam (misal '2000-01-01 00:00:00')
    # dan gagal dikenali lagi sebagai kolom periode. Makanya deteksi &
    # unpivot dilakukan di sini dulu, memakai nama kolom ASLI/mentah.
    # -----------------------------------------------------------------
    wide_period_date_col = None
    granularity, period_cols = detect_wide_period_columns(df)
    if period_cols:
        unit_hint = detect_sheet_unit_hint(file_bytes, sheet_name, header_row)
        value_col_name = f"Value ({unit_hint})" if unit_hint else "Value"
        wide_period_date_col = "Date"
        df = melt_wide_period_columns(df, period_cols, wide_period_date_col, value_col_name, granularity)
        # Baris yang nilainya kosong/'-' (ga ada transaksi tercatat di
        # periode itu) dibuang - ga ngubah hasil TOTAL apapun (NaN memang
        # diabaikan saat dijumlah/dirata-rata), tapi bikin dataset jauh
        # lebih ringkas & dashboard tetap responsif walau sheet aslinya
        # punya ratusan kolom periode.
        df[value_col_name] = clean_numeric_string(df[value_col_name], assume_dot_comma_as_thousands)
        df = df[df[value_col_name].notna()].reset_index(drop=True)

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

    # Kalau ada kolom tanggal hasil unpivot wide-format, itu yang PALING
    # merepresentasikan waktu observasi sebenarnya (bukan kolom tanggal
    # metadata lain yang mungkin ada di file, misal 'First'/'Last' yang
    # cuma nunjukkin rentang ketersediaan data) - jadi diprioritaskan
    # jadi date_cols[0] supaya turunan Year/Month otomatis ngikut ini.
    if wide_period_date_col and wide_period_date_col in date_cols:
        date_cols = [wide_period_date_col] + [c for c in date_cols if c != wide_period_date_col]

    year_cols = detect_year_columns(df)
    month_cols = detect_month_columns(df)

    # Jika hasil unpivot-nya granularity 'year' (sheet tahunan, kolom
    # 'Date' isinya angka tahun murni bukan tanggal lengkap), pastikan
    # kolom itu diprioritaskan juga sebagai year_cols[0] - supaya Year
    # Filter otomatis ngikut kolom hasil unpivot, bukan kolom lain
    # (misal 'First'/'Last' - Year) yang cuma metadata ketersediaan data.
    if wide_period_date_col and granularity == "year" and wide_period_date_col in df.columns:
        if wide_period_date_col not in year_cols:
            year_cols = [wide_period_date_col] + year_cols
        else:
            year_cols = [wide_period_date_col] + [c for c in year_cols if c != wide_period_date_col]

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

    # all_text_cols: SEMUA kolom teks TANPA batas CATEGORY_MAX_UNIQUE, dipakai
    # buat dropdown single-select di Category Analysis/Distribution/Group
    # Comparison/Kelompokkan Nilai/Summary, supaya kolom dengan unique value
    # sangat banyak (misal Vessel Name, Customer Name) TETAP muncul di
    # pilihan itu, walau nggak muncul sebagai checkbox di sidebar Category
    # Filter (yang emang sengaja dibatasi CATEGORY_MAX_UNIQUE).
    all_text_cols = detect_all_text_columns(df, exclude=list(exclude_from_category))

    meta = {
        "date_cols": date_cols,
        "year_cols": year_cols,
        "month_cols": month_cols,
        "numeric_cols": numeric_cols,
        "category_cols": category_cols,
        "all_text_cols": all_text_cols,
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
    # PENTING: key di sini diberi suffix nama sheet (f"...{selected_sheet}")
    # supaya widget-nya RESET tiap ganti sheet. Tanpa key unik ini, Streamlit
    # menganggap widget yang sama terus dan nilai "value=" default TIDAK
    # ke-apply lagi setelah interaksi pertama - jadi kalau kamu pindah sheet,
    # angka header row-nya "nyangkut" ke nilai sheet sebelumnya, dan bisa
    # bikin kolom-kolom sheet yang baru salah baca / kelihatan hilang.
    header_row = st.number_input(
        "Baris ke berapa (mulai dari 0) yang jadi header?",
        min_value=0, max_value=100, value=guessed_header_row, step=1,
        key=f"header_row_input_{selected_sheet}",
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
all_text_cols = meta["all_text_cols"]
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
    all_text_cols = [c for c in all_text_cols if c not in hidden_cols]
    id_like_cols = [c for c in id_like_cols if c not in hidden_cols]

# -----------------------------------------------------------------
# Panel debug: tunjukkan hasil deteksi kolom apa adanya, supaya user
# bisa cek langsung kalau ada kolom yang tidak terbaca sesuai harapan.
# -----------------------------------------------------------------
all_detected = set(date_cols) | set(year_cols) | set(month_cols) | set(numeric_cols) | set(all_text_cols)
undetected_cols = [c for c in df_raw.columns if c not in all_detected]

with st.expander("🔍 Kolom Terdeteksi dari File Ini (klik untuk cek)", expanded=False):
    if serial_cols:
        st.info(
            f"ℹ️ Kolom nomor urut murni terdeteksi: **{', '.join(serial_cols)}** — "
            f"kolom ini DIABAIKAN saat cek baris duplikat, supaya baris yang datanya "
            f"sama persis (cuma beda nomor urut) tetap kehitung sebagai duplikat."
        )
    text_cols_over_limit = [c for c in all_text_cols if c not in category_cols]
    if text_cols_over_limit:
        st.info(
            f"ℹ️ Kolom teks dengan unique value > {CATEGORY_MAX_UNIQUE} terdeteksi: "
            f"**{', '.join(text_cols_over_limit)}** — kolom ini TIDAK muncul sebagai "
            f"checkbox di sidebar Category Filter (biar sidebar nggak kebanjiran), "
            f"tapi TETAP tersedia penuh di dropdown Category Analysis, Distribution, "
            f"Group Comparison, Kelompokkan Nilai, dan Summary breakdown."
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
    d3.markdown("**🏷️ Text/Category Columns** (semua, termasuk unique value banyak)")
    d3.write(all_text_cols if all_text_cols else "-")
    if undetected_cols:
        st.markdown("**⚠️ Belum masuk kategori manapun** (kemungkinan tipe data campur):")
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
            options=["(Tidak dipakai)"] + all_text_cols,
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

            other_cols_for_target = [c for c in all_text_cols if c != group_source_col]
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
        if new_col_name not in all_text_cols:
            all_text_cols.append(new_col_name)

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
                    "Pilih Tahun", options=year_values, default=year_values,
                    # Key diikat ke ISI datanya - supaya kalau ganti file
                    # (range tahunnya beda), widget otomatis reset ke
                    # default penuh, bukan nyangkut pilihan file lama yang
                    # bisa bikin semua baris ke-filter habis.
                    key=f"year_filter_{year_col}_{len(year_values)}_{year_values[0]}_{year_values[-1]}",
                )
            if selected_years:
                df_filtered = df_filtered[df_filtered[year_col].isin(selected_years)]

    # -----------------------------------------------------------------
    # A. TIME FILTER - Month (robust: dukung nama bulan Indonesia/Inggris,
    # singkatan, angka 1-12, DAN kolom yang isinya tanggal lengkap seperti
    # '2024-01-01' - otomatis diambil bulannya saja tanpa bug).
    # -----------------------------------------------------------------
    if month_cols:
        month_col = month_cols[0]
        raw_month_values = df_raw[month_col].dropna().unique().tolist()

        def _to_month_name(v):
            num = month_to_number(v)
            if pd.isna(num):
                return None
            return MONTH_ORDER[int(num) - 1]

        # display_name: nama bulan standar (Januari, Februari, dst) kalau
        # nilainya kekenali (termasuk singkatan "Jan"/"Feb"/"Mar" versi
        # Inggris maupun Indonesia, atau bahkan tanggal penuh yang otomatis
        # diambil bulannya). Kalau ga kekenali formatnya, PAKAI NILAI
        # ASLINYA APA ADANYA sebagai nama tampilan - supaya SEMUA nilai
        # selalu muncul di filter, ga ada yang hilang/ke-skip.
        display_map = {v: (_to_month_name(v) or str(v)) for v in raw_month_values}

        def _sort_key(v):
            name = display_map[v]
            if name in MONTH_ORDER:
                return (0, MONTH_ORDER.index(name))
            return (1, name)

        raw_sorted = sorted(raw_month_values, key=_sort_key)
        options_display = []
        seen_display = set()
        for v in raw_sorted:
            d = display_map[v]
            if d not in seen_display:
                seen_display.add(d)
                options_display.append(d)

        if options_display:
            with st.expander("🗓️ Month Filter", expanded=True):
                selected_months_display = st.multiselect(
                    "Pilih Bulan", options=options_display, default=options_display,
                    # Key diikat ke ISI datanya (jumlah opsi + opsi pertama) -
                    # supaya kalau ganti file Excel (bulan yang muncul beda),
                    # widget otomatis reset ke default penuh, BUKAN nyangkut
                    # pilihan lama dari file sebelumnya yang bisa bikin
                    # semua baris ke-filter habis ("Tidak ada data yang cocok").
                    key=f"month_filter_{month_col}_{len(options_display)}_{options_display[0]}",
                )
            if selected_months_display:
                allowed_raw_vals = [v for v in raw_month_values if display_map[v] in selected_months_display]
                df_filtered = df_filtered[df_filtered[month_col].isin(allowed_raw_vals)]

    # -----------------------------------------------------------------
    # B. CATEGORY FILTER (checkbox) - SENGAJA tetap pakai category_cols
    # yang dibatasi CATEGORY_MAX_UNIQUE, biar sidebar nggak kebanjiran
    # ratusan/ribuan checkbox kalau ada kolom dengan unique value banyak
    # (misal Vessel Name). Kolom seperti itu tetap bisa dipakai penuh di
    # dropdown lain (Category Analysis, Distribution, dst) via all_text_cols.
    # -----------------------------------------------------------------
    if category_cols:
        with st.expander("🏷️ Category Filter", expanded=True):
            for col in category_cols[:MAX_CATEGORY_FILTERS]:
                options = sorted(df_raw[col].dropna().astype(str).unique().tolist())
                if len(options) == 0:
                    continue
                selected_vals = st.multiselect(
                    f"{col}", options=options, default=options,
                    key=f"cat_filter_{col}_{len(options)}",
                )
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
                    # Key diikat ke min/max ASLI data ini - supaya kalau
                    # user ganti file Excel (range angkanya beda), slider
                    # otomatis reset ke default penuh, BUKAN nyangkut nilai
                    # lama yang bisa di luar range baru (yang bikin semua
                    # baris ke-filter habis tanpa disadari).
                    key=f"num_filter_{col}_{round(min_v, 6)}_{round(max_v, 6)}",
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
                preview_color_map, preview_pull_cats, preview_pull_amt, preview_rotation = pick_pie_colors_and_pull(
                    preview_data_scaled[grouped_col].astype(str).tolist(), section_key=f"preview_{grouped_col}"
                )
                fig_preview = px.pie(
                    preview_data_scaled, names=grouped_col, values="Value", color=grouped_col,
                    color_discrete_map=preview_color_map,
                    title=f"{preview_label} per {grouped_col}",
                )
                fig_preview.update_traces(
                    texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
                    hovertemplate="%{label}<br>Persentase: %{percent:.2%}<br>Nilai: %{value:,.2~f}<extra></extra>",
                )
                apply_pie_customization(
                    fig_preview, preview_data_scaled, grouped_col, preview_color_map,
                    preview_pull_cats, preview_pull_amt, preview_rotation,
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

total_unique_categories = sum(df_filtered[c].nunique(dropna=True) for c in all_text_cols) \
    if all_text_cols else 0
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
    # Opsi breakdown: SEMUA kolom teks (all_text_cols, tanpa batas unique
    # value) DIGABUNG dengan kolom ID/reference number (Contract Ref.No,
    # Shipment No., dll) yang meski bertipe numeric, secara makna adalah
    # identifier -> tetap perlu bisa dipakai buat breakdown/total per
    # kelompok, termasuk kolom yang unique value-nya banyak (Vessel Name dst).
    breakdown_options = all_text_cols + [c for c in id_like_cols if c not in all_text_cols]

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
            help="Bisa pilih kolom kategori apapun (Company, Country, Vessel Name, dst) ATAU "
                 "kolom ID/reference seperti Contract Ref.No — cocok kalau 1 nilai kolom itu "
                 "muncul di banyak baris.",
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

tabs = st.tabs(["🏷️ Category", "📉 Distribution", "🔢 Numeric", "🔗 Relationship", "⚖️ Group Comparison"])

# -----------------------------------------------------------------
# A. CATEGORY ANALYSIS
# -----------------------------------------------------------------
with tabs[0]:
    if all_text_cols and numeric_cols:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            cat_col = st.selectbox("Pilih Category", all_text_cols, key="cat_col_bar")
        with cc2:
            val_col = st.selectbox("Pilih Numeric", numeric_cols, key="cat_val_bar")
        with cc3:
            top_n = st.selectbox("Tampilkan Top", options=[1, 3, 5, 10, 15, 20], index=3, key="top_n_bar")

        cat_agg = df_filtered.groupby(cat_col, as_index=False)[val_col].sum()
        cat_agg = cat_agg.sort_values(val_col, ascending=False)
        cat_agg_scaled, cat_unit_caption = scale_for_display(cat_agg, val_col, val_col)

        b1, b2 = st.columns(2)
        with b1:
            # Judul default SAMA seperti sebelumnya ("{cat_col} vs Total {val_col}"),
            # tapi sekarang bisa diedit bebas lewat kotak input di bawah.
            default_title_left = f"{cat_col} vs Total {val_col}"
            edited_title_left = st.text_input(
                "✏️ Judul Chart (bisa diedit)", value=default_title_left,
                key=f"chart_title_left_{cat_col}_{val_col}",
                help="Judul otomatis ke-generate dari kolom yang dipilih, tapi bisa diubah bebas sesuai kebutuhan.",
            )
            fig_bar = px.bar(
                cat_agg_scaled, x=cat_col, y=val_col,
                title=edited_title_left or default_title_left,
            )
            show_chart(fig_bar, n_categories=len(cat_agg), unit_caption=cat_unit_caption)
        with b2:
            top_n_data = cat_agg.head(top_n).sort_values(val_col)
            top_n_data_scaled, top_n_unit_caption = scale_for_display(top_n_data, val_col, val_col)
            # Judul default SAMA seperti sebelumnya ("Top N {cat_col} by {val_col}"),
            # juga bisa diedit bebas.
            default_title_right = f"Top {top_n} {cat_col} by {val_col}"
            edited_title_right = st.text_input(
                "✏️ Judul Chart (bisa diedit)", value=default_title_right,
                key=f"chart_title_right_{cat_col}_{val_col}_{top_n}",
                help="Judul otomatis ke-generate dari kolom yang dipilih, tapi bisa diubah bebas sesuai kebutuhan.",
            )
            fig_hbar = px.bar(
                top_n_data_scaled, x=val_col, y=cat_col, orientation="h",
                title=edited_title_right or default_title_right,
                text=val_col,
            )
            fig_hbar.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")
            show_chart(fig_hbar, n_categories=len(top_n_data), unit_caption=top_n_unit_caption)
    else:
        st.info("Tidak ditemukan kombinasi kolom category dan numeric untuk Category Analysis.")

# -----------------------------------------------------------------
# B. DISTRIBUTION
# -----------------------------------------------------------------
with tabs[1]:
    if all_text_cols:
        dc1, dc2 = st.columns(2)
        with dc1:
            dist_col = st.selectbox("Pilih Category untuk Distribusi", all_text_cols, key="dist_col")
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
        dist_color_map, dist_pull_cats, dist_pull_amt, dist_rotation = pick_pie_colors_and_pull(
            dist_data[dist_col].astype(str).tolist(), section_key=f"dist_{dist_col}"
        )
        render_full_vs_others_pies(
            dist_data, dist_col, "Value", f"Distribution of {dist_col} ({value_label})", others_selected_dist,
            unit_source_col=dist_unit_source,
            color_map=dist_color_map, pull_categories=dist_pull_cats,
            pull_amount=dist_pull_amt, rotation_deg=dist_rotation,
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
                    help="Pilih kategori mana saja yang mau digabung jadi satu slice 'Others'.",
                )
                drill_color_map, drill_pull_cats, drill_pull_amt, drill_rotation = pick_pie_colors_and_pull(
                    drill_data[drill_src_col].astype(str).tolist(),
                    section_key=f"drill_{selected_drill_group}_{drill_src_col}",
                )
                render_full_vs_others_pies(
                    drill_data, drill_src_col, "Value",
                    f"Rincian '{selected_drill_group}' per {drill_src_col} ({value_label})",
                    others_selected_drill,
                    unit_source_col=dist_unit_source,
                    color_map=drill_color_map, pull_categories=drill_pull_cats,
                    pull_amount=drill_pull_amt, rotation_deg=drill_rotation,
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
# C. NUMERIC ANALYSIS
# -----------------------------------------------------------------
with tabs[2]:
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
# D. RELATIONSHIP
# -----------------------------------------------------------------
with tabs[3]:
    if len(numeric_cols) >= 2:
        r1, r2 = st.columns(2)
        with r1:
            x_col = st.selectbox("Sumbu X", numeric_cols, index=0, key="scatter_x")
        with r2:
            y_default_idx = 1 if len(numeric_cols) > 1 else 0
            y_col = st.selectbox("Sumbu Y", numeric_cols, index=y_default_idx, key="scatter_y")

        color_arg = all_text_cols[0] if all_text_cols else None
        fig_scatter = px.scatter(
            df_filtered, x=x_col, y=y_col, color=color_arg,
            title=f"{x_col} vs {y_col}",
        )
        show_chart(fig_scatter)
    else:
        st.info("Minimal dibutuhkan 2 kolom numeric untuk membuat Scatter Plot.")

# -----------------------------------------------------------------
# E. GROUP COMPARISON (generic - berlaku untuk kolom & nilai apapun,
# tidak terikat ke istilah/kasus bisnis tertentu)
# -----------------------------------------------------------------
with tabs[4]:
    st.caption(
        "Bagi data jadi 2 kelompok berdasarkan kolom & nilai pilihan kamu sendiri "
        "(misal Domestik vs Ekspor, Aktif vs Nonaktif, atau apapun sesuai isi file), "
        "lalu breakdown salah satu kelompoknya lebih detail per sub-kategori dengan warna berbeda."
    )
    if all_text_cols and numeric_cols:
        gc1, gc2 = st.columns(2)
        with gc1:
            group_class_col = st.selectbox(
                "Kolom Klasifikasi (untuk membagi 2 kelompok)",
                options=all_text_cols,
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

        other_category_cols = [c for c in all_text_cols if c != group_class_col]
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

                gcomp_color_map, gcomp_pull_cats, gcomp_pull_amt, gcomp_rotation = pick_pie_colors_and_pull(
                    group_breakdown[group_sub_col].astype(str).tolist(),
                    section_key=f"gcomp_{group_a_name}_{group_sub_col}",
                )
                render_full_vs_others_pies(
                    group_breakdown, group_sub_col, group_value_col,
                    f"Distribusi {group_a_name} per {group_sub_col}",
                    others_selected_comparison,
                    unit_source_col=group_value_col,
                    color_map=gcomp_color_map, pull_categories=gcomp_pull_cats,
                    pull_amount=gcomp_pull_amt, rotation_deg=gcomp_rotation,
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
