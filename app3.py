"""
=====================================================================
DYNAMIC EXCEL DASHBOARD - GENERIC AUTO-ANALYTICS ENGINE
=====================================================================
Generic Streamlit-based dashboard that reads any Excel file (no
hardcoded dataset), automatically detects the data structure
(numeric, date, year, month, category), and builds filters,
KPIs, visualizations, and export of the filtered results.

How to run:
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
# Page Config (must be called first thing in the script)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Excel Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Custom CSS so the dashboard looks compact & professional
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

# Indonesian month names are also recognized ('Januari', 'Februari', etc,
# not just English month names) - lots of Indonesian Excel files use these,
# which used to make the Month column get detected but its filter would
# end up empty/not show up.
MONTH_ALIAS_ID = [
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
]
MONTH_ALIAS.update({m: i + 1 for i, m in enumerate(MONTH_ALIAS_ID)})
MONTH_ALIAS.update({m[:3]: i + 1 for i, m in enumerate(MONTH_ALIAS_ID)})
# Common abbreviations that differ from the standard 3-letter form
MONTH_ALIAS.update({"agt": 8, "ags": 8, "des": 12, "nov": 11, "okt": 10})

CATEGORY_MAX_UNIQUE = 200
MAX_NUMERIC_SLIDERS = 20
MAX_CATEGORY_FILTERS = 20


# =====================================================================
# PART 2: DATA CLEANING & AUTO DETECTION ENGINE
# =====================================================================

def is_text_dtype(series: pd.Series) -> bool:
    """True for text columns, compatible with the classic 'object' dtype
    as well as the newer 'string'/StringDtype (pandas >= 2.x/3.x), but
    still False for numeric/datetime/bool columns."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return False
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names: strip whitespace, remove newlines, tidy up."""
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
    """Drop columns that are entirely empty (all NaN) or empty 'unnamed' columns."""
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    for c in unnamed_cols:
        if df[c].isna().all():
            df = df.drop(columns=[c])
    return df


def clean_numeric_string(series: pd.Series, assume_single_separator_as_thousands: bool = False) -> pd.Series:
    """Clean a numeric string into a number EXACTLY, without rounding.

    Cases that are NOT ambiguous (always processed automatically, safe):
    - Both a dot AND a comma are present -> whichever appears LAST is treated
      as the decimal separator, the rest is treated as a thousands separator.
      Example: "8.847,50" -> 8847.5, "8,847.50" -> 8847.5
    - More than one dot (no comma) -> definitely a thousands separator.
      Example: "1.234.567" -> 1234567
    - More than one comma (no dot) -> definitely a thousands separator.
      Example: "1,234,567" -> 1234567

    AMBIGUOUS cases (only a single dot OR a single comma, e.g. "8.847" or
    "8,847") DEFAULT to being treated as a GENUINE DECIMAL (the dot is left
    as-is, the comma is converted to a dot) so that NO number ever changes
    without the user knowing. If assume_single_separator_as_thousands=True,
    it will instead try to guess it as a thousands separator (only when the
    pattern is exactly 3 digits after the symbol).
    """
    def _clean(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val
        s = str(val).strip()
        if s == "":
            return np.nan

        # handle negative numbers in parentheses, e.g. (1000)
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1]

        # remove currency symbols & spaces (NOT comma/dot - those are handled separately below)
        s = re.sub(r"[Rp$€£¥\s]", "", s)

        # handle percent
        is_percent = s.endswith("%")
        s = s.replace("%", "")

        has_comma = "," in s
        has_dot = "." in s

        if has_comma and has_dot:
            # Not ambiguous: whichever appears last = decimal
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif has_dot and not has_comma:
            if s.count(".") > 1:
                # Not ambiguous: multiple dots = definitely a thousands separator, e.g. "1.234.567"
                s = s.replace(".", "")
            elif assume_single_separator_as_thousands:
                int_part, frac_part = s.split(".")
                if len(frac_part) == 3 and int_part not in ("", "0") and len(int_part) <= 3:
                    s = s.replace(".", "")
                # else leave it as a decimal
            # default: leave as-is (genuine decimal), e.g. "8.847" stays 8.847
        elif has_comma and not has_dot:
            if s.count(",") > 1:
                # Not ambiguous: multiple commas = definitely a thousands separator
                s = s.replace(",", "")
            elif assume_single_separator_as_thousands:
                int_part, frac_part = s.split(",")
                if len(frac_part) == 3 and int_part not in ("", "0") and len(int_part) <= 3:
                    s = s.replace(",", "")
                else:
                    s = s.replace(",", ".")
            else:
                # default: single comma treated as decimal, e.g. "8,5" -> 8.5
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
    """Try converting object columns into numeric if the majority succeeds."""
    df = df.copy()
    for col in df.columns:
        if is_text_dtype(df[col]):
            cleaned = clean_numeric_string(df[col], assume_dot_comma_as_thousands)
            valid_ratio = cleaned.notna().sum() / max(len(cleaned), 1)
            original_non_null = df[col].notna().sum()
            if original_non_null > 0 and valid_ratio >= 0.7 and cleaned.notna().sum() > 0:
                df[col] = cleaned
    return df


# Minimum number of date-named columns required to be considered a genuine
# "wide time-series" (not just a coincidental 1-2 columns named like a date).
WIDE_PERIOD_MIN_COLS = 6


def detect_wide_period_columns(df: pd.DataFrame):
    """Detect columns whose NAME itself is a date/period (e.g. column
    headers '2000-01-01', '2000-02-01', etc - one column per month/period/
    year). This is a common WIDE format used in trade/statistics reports
    (e.g. Global Trade Tracker, national statistics bureaus, export-import
    data): each row = 1 category combination, each column = 1 point in
    time, holding a number. Supports 2 granularities:
    - 'date': column headers are full dates (e.g. a monthly sheet)
    - 'year': column headers are pure year numbers (e.g. an annual sheet,
      headers like '2000', '2001', etc - with NO month info)
    If there are many such columns (>= WIDE_PERIOD_MIN_COLS), the dashboard
    will automatically 'unpivot' (melt) them so they can be analyzed with
    the same date/year/trend filter features as regular long-format data -
    generic, works for any Excel sheet with this pattern, not just one
    specific file.
    Returns (granularity, period_cols) - granularity is None if not detected."""
    date_period_cols = []
    year_period_cols = []
    for col in df.columns:
        if isinstance(col, (pd.Timestamp, datetime.date, datetime.datetime)):
            date_period_cols.append(col)
            continue
        col_str = str(col).strip()
        # Matches only if the column name TRULY looks like a date pattern
        # (not just any number) - e.g. '2000-01-01', '2000-01', 'Jan-2000',
        # 'Jan 2000'. Tolerant of a time component (e.g. '2000-01-01
        # 00:00:00' or '2000-01-01T00:00:00') so header columns that were
        # originally datetime but got coerced to text (e.g. via
        # str()/to_string()) are still detected correctly, not just ones
        # that are still raw datetime objects.
        looks_like_date = bool(
            re.match(r"^\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$", col_str)
            or re.match(r"^[A-Za-z]{3,9}[\s\-]\d{2,4}$", col_str)
        )
        if looks_like_date:
            parsed = pd.to_datetime(col_str, errors="coerce")
            if pd.notna(parsed):
                date_period_cols.append(col)
                continue
        # Check for a PURE YEAR pattern (e.g. a header that's the number
        # 2000, 2000.0, or the string '2000') - used when the sheet is a
        # wide-format annual sheet.
        year_val = None
        if isinstance(col, (int, float, np.integer, np.floating)) and not isinstance(col, bool):
            if float(col).is_integer():
                year_val = int(col)
        elif re.fullmatch(r"(19|20)\d{2}", col_str):
            year_val = int(col_str)
        elif re.fullmatch(r"(19|20)\d{2}\.0", col_str):
            # A year header that's a float which already got str()-ified,
            # e.g. '2000.0' (from an Excel column of dtype float 2000.0).
            year_val = int(float(col_str))
        if year_val is not None and 1900 <= year_val <= 2100:
            year_period_cols.append(col)

    if len(date_period_cols) >= WIDE_PERIOD_MIN_COLS:
        return "date", date_period_cols
    if len(year_period_cols) >= WIDE_PERIOD_MIN_COLS:
        return "year", year_period_cols
    return None, []


def detect_sheet_unit_hint(file_bytes: bytes, sheet_name, header_row: int) -> str:
    """Try to find a unit hint (e.g. '(Tonnes)') from the rows ABOVE the
    header row - many trade/statistics reports write the unit there,
    separate from the column names. Returns '' if nothing is found."""
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
    """Turn period columns (WIDE format, 1 column = 1 point in time) into
    LONG format: 1 row = 1 combination of id + period + value. Other id
    columns (e.g. 'Import country', 'First', 'Last') are kept as-is. After
    melting, the dashboard can immediately use the date/month/year filter
    and trend visualization features just like regular long-format data.
    `granularity`: 'date' (resulting column becomes a full datetime) or
    'year' (resulting column becomes a plain year number, for annual sheets)."""
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
    """Detect columns where the majority of values can be converted to datetime."""
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
                # IMPORTANT: pandas.to_datetime mis-parses plain month names
                # (e.g. 'Jan', 'Feb', 'December') into a date with a DEFAULT
                # year of 0001 (since there's no real year info) - e.g. 'Jan'
                # -> 0001-01-01. If all parsed results share the same year 1,
                # this is NOT a genuine date column (most likely a duplicate
                # Month column that got detected twice), so skip it from
                # date_cols.
                valid_years = converted.dropna().dt.year
                if not valid_years.empty and valid_years.nunique() == 1 and valid_years.iloc[0] == 1:
                    continue
                date_cols.append(col)
        except Exception:
            continue
    return date_cols


def detect_year_columns(df: pd.DataFrame) -> list:
    """Detect year columns: name contains 'year'/'tahun' OR values are reasonable 4-digit years."""
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
    """Detect month columns: name contains 'month'/'bulan' OR values are month names/numbers."""
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
    """Convert a month value (name, number, OR a full date) into a
    number from 1-12. If the 'Month' column turns out to actually contain
    full dates (e.g. '2024-01-01' or a Timestamp/date object), the month
    part is automatically extracted - so the filter still shows 'January'
    etc, not a raw 'yyyy-mm-dd' date."""
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
    # If the value is a number written as text, e.g. "5" or "05"
    if s.isdigit():
        v = int(s)
        return v if 1 <= v <= 12 else np.nan
    # If the value is a full date string (e.g. "2024-01-01", "01/02/2024",
    # "Jan-2024") - try parsing it as a date, then take its month.
    parsed = pd.to_datetime(val, errors="coerce")
    if pd.notna(parsed):
        return parsed.month
    return np.nan


def detect_category_columns(df: pd.DataFrame, exclude: list) -> list:
    """Detect categorical columns: object/string with a LIMITED number of
    unique values (<= CATEGORY_MAX_UNIQUE). Used SPECIFICALLY for the
    sidebar Category Filter (checkboxes), so the sidebar doesn't get
    flooded with hundreds/thousands of checkboxes when a column has a
    very large number of unique values (e.g. vessel name, customer name)."""
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
    """All text columns WITHOUT a cap on the number of unique values - used
    for the single-select dropdowns everywhere (Category Analysis,
    Distribution, Group Comparison, Group Values, Summary breakdown, etc)
    which can safely hold thousands of options since it's just a regular
    dropdown, not checkboxes. DIFFERENT from detect_category_columns, which
    is deliberately capped at CATEGORY_MAX_UNIQUE specifically for the
    sidebar Category Filter."""
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

# Column-name patterns that are LIKELY just a serial/running number (not a
# genuine business key like "Contract Ref.No", which is expected to
# legitimately repeat for the same contract). Deliberately kept NARROW
# (must match the whole name) so it doesn't get confused with an
# important identifier column.
PURE_SERIAL_NAME_PATTERN = re.compile(
    r"^(no\.?|s\/n|sn|seq|sequence|index|row\s*no\.?|nomor\s*urut)$",
    re.IGNORECASE,
)


def detect_pure_serial_columns(df: pd.DataFrame) -> list:
    """Detect PURE running-number columns (e.g. 'No.' containing
    1,2,3,4,...) whose values are almost always unique per row. Columns
    like this MUST be ignored when checking for duplicates, otherwise
    rows that are actually exact duplicates (just with a different serial
    number) will never be detected as duplicates -> totals end up wrong /
    double-counted."""
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
    """Detect numeric columns that are actually an ID/reference number
    (e.g. 'Contract Ref.No', 'Shipment No.', 'No.') -> not a number that's
    meaningful to sum/average, but still useful as a breakdown key."""
    id_cols = []
    for col in numeric_cols:
        if ID_NAME_PATTERN.search(str(col)):
            id_cols.append(col)
    return id_cols


def fmt_num(x, decimals: int = 2) -> str:
    """Format a number with thousands separators, showing AT MOST
    `decimals` digits after the decimal point - but if the number is a
    whole number / doesn't need decimals, the decimals are dropped
    (e.g. 3000000 -> '3,000,000', NOT '3,000,000.00'). If there is a
    fractional part, it still shows up to `decimals` digits
    (e.g. 1234.5 -> '1,234.50')."""
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
    """Auto-detect a unit hint from the Excel COLUMN NAME, e.g.:
    - 'Volume (Million MT)'  -> 'Million MT'
    - 'Weight in Kg'         -> 'Kg'
    - 'Sales (USD Thousand)' -> 'USD Thousand'
    - 'Revenue in Million USD' -> 'Million USD'
    This keeps the dashboard fully generic (no hardcoded column names) -
    the unit hint automatically follows whatever is written in the Excel
    header. Returns an empty string "" if no unit pattern is found."""
    if not isinstance(col_name, str):
        return ""

    # 1) Pattern in parentheses at the end of the column name: "Column Name (unit)"
    paren_matches = re.findall(r"\(([^)]+)\)", col_name)
    for candidate in reversed(paren_matches):
        candidate = candidate.strip()
        # the unit must contain letters (not just some unclear code/number)
        # and be short enough that it's not accidentally a long sentence
        if re.search(r"[A-Za-z]", candidate) and len(candidate) <= 40:
            return candidate

    # 2) "in <unit>" pattern at the end of the column name, e.g. "Revenue in Million USD"
    in_match = re.search(r"\bin\s+([A-Za-z][A-Za-z0-9\s./%-]{1,35})$", col_name, re.IGNORECASE)
    if in_match:
        return in_match.group(1).strip()

    return ""


def show_unit_note(col_name: str) -> None:
    """Show a small caption if the column name `col_name` has an
    auto-detectable unit hint (see detect_column_unit)."""
    unit = detect_column_unit(col_name)
    if unit:
        st.caption(f"📏 Unit detected from column name **{col_name}**: *in {unit}*")


# Recognized scale words (Indonesian & English) with their multipliers -
# including common abbreviations used in commodity/trade reports (coal,
# oil, etc), e.g. 'mn t' (million ton), 'bn t' (billion ton).
SCALE_WORDS = {
    "thousand": 1_000, "ribu": 1_000, "th": 1_000, "k": 1_000,
    "million": 1_000_000, "juta": 1_000_000, "mn": 1_000_000, "mio": 1_000_000,
    "billion": 1_000_000_000, "miliar": 1_000_000_000, "milyar": 1_000_000_000,
    "bn": 1_000_000_000, "bio": 1_000_000_000,
}
# Sorted from longest to shortest, used to check COMBINED tokens with no
# space (e.g. 'mnt' = 'mn' + 't', 'bnusd' = 'bn' + 'usd'). DELIBERATELY
# only long, unambiguous abbreviations ('mn','bn','mio','bio') - single
# letters like 'k'/'th' are EXCLUDED from this combined mode since they're
# prone to clashing with already-common unit prefixes (e.g. 'kg'
# kilogram, 'km' kilometer, 'kt' kiloton) - those should be read as a
# whole unit, not split into 'thousand + g/m/t'.
_SAFE_COMBINED_PREFIXES = ["million", "billion", "thousand", "mio", "bio", "mn", "bn"]
_SCALE_PREFIXES_BY_LEN = sorted(_SAFE_COMBINED_PREFIXES, key=len, reverse=True)


def _split_scale_prefix(token: str):
    """Try to split a combined token like 'mnt' -> ('mn', 't')."""
    tl = token.lower()
    for prefix in _SCALE_PREFIXES_BY_LEN:
        if tl.startswith(prefix) and len(tl) > len(prefix):
            rest = token[len(prefix):]
            if rest.isalpha():
                return prefix, rest
    return None, None


def parse_unit_parts(unit_text: str):
    """Split a unit hint into (explicit_scale, scale_word, base_unit).
    E.g.: 'Million MT' -> (1_000_000, 'Million', 'MT')
          'MT'         -> (None, None, 'MT')
          'USD Thousand' -> (1_000, 'Thousand', 'USD')
          'mn t'       -> (1_000_000, 'mn', 't')
          'mnt'        -> (1_000_000, 'mn', 't')   (combined token, no space)
    If there's no scale word in the text, explicit_scale = None (meaning
    the data is STILL in raw/original units, not yet scaled)."""
    if not unit_text:
        return None, None, ""
    tokens = unit_text.split()
    for i, tok in enumerate(tokens):
        key = tok.lower()
        if key in SCALE_WORDS:
            rest = " ".join(tokens[:i] + tokens[i + 1:]).strip()
            return SCALE_WORDS[key], tok, rest
        # Try a combined token with no space, e.g. 'mnt' -> scale 'mn' + rest 't'
        prefix, rest_of_token = _split_scale_prefix(tok)
        if prefix:
            rest_tokens = tokens[:i] + ([rest_of_token] if rest_of_token else []) + tokens[i + 1:]
            rest = " ".join(rest_tokens).strip()
            return SCALE_WORDS[prefix], prefix, rest
    return None, None, unit_text


def auto_scale_factor(max_abs_value: float):
    """Auto-detect a divisor (Thousand/Million/Billion) based on the
    magnitude of the numbers, so the numbers shown in the chart are more
    compact and easier to read (e.g. 22,000,123 becomes 22,000 with a
    caption of 'In Th'). Small numbers (< 1,000) aren't scaled at all."""
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


# Short form used for the unit caption on charts - more presentable and
# commonly used in commodity/trade reports (e.g. 'Mn T' = Million Ton),
# rather than spelling out 'Million'/'Billion'/'Thousand' in full.
SCALE_ABBREV = {1_000: "Th", 1_000_000: "Mn", 1_000_000_000: "Bn"}
# Leading letter considered "redundant" if the base unit starts with the
# same letter as the scale already mentioned earlier in the caption -
# e.g. scale 'Million' + unit 'MT' -> the 'M' in front of 'MT' would
# double up on "Million", so it's dropped, leaving just 'T'. Only applies
# to Million/Billion - NOT Thousand, because 'KT' (kiloton) is a valid
# unit in its own right, not redundant, so it must not be trimmed.
_REDUNDANT_SCALE_LETTER = {1_000_000: "m", 1_000_000_000: "b"}


def _strip_redundant_scale_letter(base_unit: str, scale_val: int) -> str:
    """Strip the leading letter of base_unit if it duplicates the scale
    already present in the caption (e.g. 'MT' + Million scale -> 'T')."""
    letter = _REDUNDANT_SCALE_LETTER.get(scale_val)
    if not letter or not base_unit or len(base_unit) <= 1:
        return base_unit
    if base_unit[0].lower() == letter:
        return base_unit[1:]
    return base_unit


def get_display_scale(col_name: str, series: pd.Series):
    """Determine (divisor, caption) for displaying the numbers of column
    `col_name` with data `series` on a chart:
    - If the column name ALREADY explicitly states a scale (e.g.
      'Volume (Million MT)' or 'Volume (mn t)'), the data is assumed to
      ALREADY be in that scale -> not divided again (divisor=1), and the
      caption is shortened/cleaned up into 'In Mn T' (instead of the
      longer 'In Million MT').
    - If there's NO explicit scale yet (e.g. just 'Volume (MT)' or plain
      'Volume' with a large raw number like 22,000,123), the appropriate
      scale is auto-detected from the magnitude, the number is DIVIDED to
      be more compact, and the caption becomes e.g. 'In Th MT' - this is
      what turns '22,000,123' into '22,000'."""
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
    """Apply auto-scaling to `data[value_col]` based on the source column
    name `unit_source_col` (used for unit detection). Returns
    (new_scaled_data, unit_caption). If `unit_source_col` is None/empty
    (e.g. computing Row Count, not a genuine numeric column), no scaling
    is applied."""
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
    """Combine rows whose label is in `selected_values` into a single
    'Others' row - the user chooses which categories should be counted
    as 'Others', rather than it being automatic based on percentage."""
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


# =====================================================================
# CHART EDIT CONTROLS: title, unit (manual correction), and font size -
# applies UNIVERSALLY to every chart in the dashboard, not just one
# specific bar chart. The font size here is also used for the downloaded
# PNG (not just the on-screen display), because Plotly stores that font
# size directly inside the chart definition that gets exported.
# =====================================================================

DEFAULT_CHART_FONT_SIZE = 15


def chart_controls(default_title: str, auto_unit_caption: str = "", key_prefix: str = "chart",
                    show_unit: bool = True) -> tuple:
    """Render a small expander with 3 universal controls for a chart:
    - Title (editable, defaults to a name derived from the selected column)
    - Unit hint / subtitle (editable - IMPORTANT for manually correcting
      the auto-detected unit from the Excel column name if it's wrong/
      not quite right; leave blank to show nothing)
    - Font size (affects both the on-screen display AND the downloaded
      PNG - increase it if the text in the downloaded file is too small)
    Returns (final_title, final_unit_caption, font_size)."""
    with st.expander("✏️ Edit Chart (Title / Unit / Font Size)", expanded=False):
        c1, c2 = st.columns(2) if show_unit else (st.container(), None)
        with c1:
            title = st.text_input(
                "Chart Title", value=default_title, key=f"{key_prefix}_title_ctrl",
                help="Auto-filled from the selected column, but can be changed freely as needed.",
            )
        unit_caption = auto_unit_caption
        if show_unit:
            with c2:
                unit_caption = st.text_input(
                    "Unit Hint (subtitle below the title)",
                    value=auto_unit_caption, key=f"{key_prefix}_unit_ctrl",
                    help="Auto-detected from the Excel column name, but sometimes it's wrong/not "
                         "quite right - edit it manually here if needed. Leave blank to hide it.",
                )
        font_size = st.slider(
            "Font Size (applies on-screen & in the downloaded file)",
            min_value=10, max_value=32, value=DEFAULT_CHART_FONT_SIZE, step=1,
            key=f"{key_prefix}_font_ctrl",
            help="This size is also used for the downloaded image (the camera button on the "
                 "chart) - increase it if the text in the downloaded file is too small.",
        )
    return (title.strip() or default_title), unit_caption, font_size


def show_chart(
    fig, n_categories: int = None, is_pie: bool = False, base_height: int = 460, unit_caption: str = "",
    enable_selection: bool = False, chart_key: str = None, font_size: int = DEFAULT_CHART_FONT_SIZE,
):
    """Display a plotly chart with consistent styling so that:
    1. Font size CAN BE ADJUSTED by the user via `font_size` (from
       chart_controls()) - used both for the on-screen display AND the
       downloaded image, since Plotly stores that font size directly in
       the chart definition that gets exported to PNG.
    2. Labels/text (category names, numbers, legend) are NEVER CUT OFF -
       uses automargin, more generous margins, and the chart height
       automatically grows when there are many categories.
    3. When downloaded (the camera icon in the top-right corner of the
       chart), the resulting image is high-resolution & sharp.
    `n_categories`: number of categories/slices/bars, used to decide the
    chart's height & whether the X-axis labels need to be tilted to avoid
    overlapping.
    `unit_caption`: if provided (e.g. 'In Mn T'), it's combined into a
    neat SUBTITLE below the chart title.
    If `enable_selection=True`, returns the chart's selection/click event
    (used for the cross-filtering feature) - otherwise returns None."""
    title_size = font_size + 4
    tick_size = max(9, font_size - 2)
    legend_size = max(9, font_size - 2)

    fig.update_layout(
        font=dict(size=font_size),
        title_font=dict(size=title_size),
        legend=dict(font=dict(size=legend_size)),
        margin=dict(l=70, r=50, t=70, b=90),
        uniformtext_minsize=max(8, font_size - 3),
        uniformtext_mode="show",
    )
    fig.update_xaxes(automargin=True, tickfont=dict(size=tick_size))
    fig.update_yaxes(automargin=True, tickfont=dict(size=tick_size))

    height = base_height
    if n_categories:
        # the more categories there are, the taller the chart gets so
        # labels & the legend don't pile up / get cut off
        height = max(base_height, base_height + (n_categories - 6) * 22)
        if n_categories > 8 and not is_pie:
            fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=height)

    if is_pie:
        fig.update_traces(textfont=dict(size=tick_size))
        # This is KEY for outside pie labels getting cut off: automargin
        # makes plotly automatically widen the figure's margins so that
        # text labels outside the slice (category name + percent + value)
        # always fit, and never get clipped at the chart's edge.
        fig.update_traces(automargin=True, selector=dict(type="pie"))
        fig.update_layout(margin=dict(l=90, r=90, t=90, b=90))

    if unit_caption:
        # A proper subtitle below the title (not a separate annotation) -
        # so it always looks neat & never overlaps the title regardless
        # of how long the title is.
        current_title = ""
        if fig.layout.title and fig.layout.title.text:
            current_title = fig.layout.title.text
        height += 26
        subtitle_size = max(8, font_size - 2)
        fig.update_layout(
            title=dict(
                text=f"{current_title}<br><span style='font-size:{subtitle_size}px;color:#666666'>{unit_caption}</span>",
            ),
            height=height,
            margin=dict(t=(fig.layout.margin.t or 70) + 26),
        )

    plotly_kwargs = dict(
        use_container_width=True,
        config={
            "displaylogo": False,
            # Size & scale of the downloaded image is increased so the
            # text stays clearly readable - the font_size above also
            # gets applied here since it's already baked into the `fig`
            # layout.
            "toImageButtonOptions": {
                "format": "png",
                "filename": "chart",
                "height": max(900, height * 2),
                "width": 1600,
                "scale": 3,
            },
        },
    )
    if enable_selection:
        # on_select="rerun" makes clicking/selecting on this chart trigger
        # a Streamlit rerun with the clicked point's info stored in the
        # return value - this is the basis for the cross-filtering
        # feature (click a bar/slice, other parts of the page filter
        # automatically).
        event = st.plotly_chart(fig, on_select="rerun", key=chart_key, **plotly_kwargs)
        return event
    st.plotly_chart(fig, **plotly_kwargs)
    return None


PIE_DEFAULT_PALETTE = px.colors.qualitative.Plotly


def pick_pie_colors_and_pull(categories: list, section_key: str):
    """Show interactive controls (in an expander) for customizing a pie
    chart: custom color per category, choosing which slice(s) to "pull
    out" (explode), and the chart's starting rotation. `section_key` must
    be unique per call (e.g. tab name + column name) so the remembered
    widgets & colors don't collide/overwrite another pie chart in a
    different tab.
    Returns: (color_map: dict, pull_categories: list, pull_amount: float, rotation_deg: int)
    """
    categories = [str(c) for c in categories]
    color_map = {}
    with st.expander(f"🎨 Customize Colors & Explode ({section_key})"):
        st.caption(
            "Pick your own color per category, pull out (explode) a specific slice to "
            "highlight it, and adjust the chart's starting rotation if needed."
        )
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            st.markdown("**Color per category**")
            color_grid = st.columns(3)
            for i, cat in enumerate(categories):
                default_color = PIE_DEFAULT_PALETTE[i % len(PIE_DEFAULT_PALETTE)]
                widget_key = f"color_{section_key}_{cat}"
                chosen = color_grid[i % 3].color_picker(cat, value=default_color, key=widget_key)
                color_map[cat] = chosen
        with cc2:
            st.markdown("**Explode & Rotation**")
            pull_categories = st.multiselect(
                "Pull out (explode) this slice",
                options=categories,
                key=f"pull_cats_{section_key}",
                help="The selected slice(s) will be pulled slightly outward from the circle to highlight the focus.",
            )
            pull_amount = 0.0
            if pull_categories:
                pull_amount = st.slider(
                    "Pull-out amount", min_value=0.02, max_value=0.4, value=0.12, step=0.02,
                    key=f"pull_amt_{section_key}",
                )
            rotation_deg = st.slider(
                "Starting rotation (degrees)", min_value=0, max_value=360, value=0, step=15,
                key=f"rotation_{section_key}",
                help="Changes the position of the first slice (12 o'clock = 0 degrees). Purely "
                     "cosmetic, does not change the data.",
            )
    return color_map, pull_categories, pull_amount, rotation_deg


def apply_pie_customization(fig, data: pd.DataFrame, label_col: str, color_map: dict = None,
                             pull_categories: list = None, pull_amount: float = 0.0,
                             rotation_deg: int = 0) -> None:
    """Apply pull (explode) & rotation to an already-created px.pie trace.
    Colors are already applied via color_discrete_map when px.pie is
    called, so this function focuses on pull & rotation, which need the
    original data order."""
    pull_categories = pull_categories or []
    if pull_categories and pull_amount:
        pull_array = [pull_amount if str(v) in pull_categories else 0 for v in data[label_col]]
        fig.update_traces(pull=pull_array)
    if rotation_deg:
        fig.update_traces(rotation=rotation_deg)


def render_full_vs_others_pies(
    data: pd.DataFrame, label_col: str, value_col: str, title_prefix: str, others_selected: list,
    unit_source_col: str = None, color_map: dict = None, pull_categories: list = None,
    pull_amount: float = 0.0, rotation_deg: int = 0, key_prefix: str = "pie",
) -> None:
    """Show 2 comparison pie charts: left/top is the FULL version (all
    categories as-is), right/bottom is the version with selected
    categories combined into 'Others'. If there are many categories,
    they're stacked VERTICALLY (full width) instead of side-by-side, so
    outside labels have more room and don't get cut off.
    `unit_source_col`: the name of the ORIGINAL numeric column (for unit
    detection & auto-scaling large numbers, e.g. 22,000,123 -> 22,000 +
    caption 'In Th MT'). Pass None if value_col isn't a genuine numeric
    column (e.g. computing Row Count).
    Each pie (Full & Others) has ITS OWN INDEPENDENT controls (title,
    unit, font size) via chart_controls(), so both charts can be adjusted
    independently of each other."""
    data, auto_unit_caption = scale_for_display(data, value_col, unit_source_col)
    data_with_others = group_selected_as_others(data, label_col, value_col, others_selected)
    n_max = max(len(data), len(data_with_others))
    stack_vertically = n_max > 10

    if stack_vertically:
        pc_full = pc_others = st.container()
    else:
        pc_full, pc_others = st.columns(2)

    with pc_full:
        st.caption("📊 Full (all categories)")
        title_full, unit_full, font_full = chart_controls(
            default_title=f"{title_prefix} - Full", auto_unit_caption=auto_unit_caption,
            key_prefix=f"{key_prefix}_full",
        )
        fig_full = px.pie(
            data, names=label_col, values=value_col, title=title_full,
            color=label_col, color_discrete_map=color_map,
        )
        fig_full.update_traces(
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
            hovertemplate="%{label}<br>Percentage: %{percent:.2%}<br>Value: %{value:,.2~f}<extra></extra>",
        )
        apply_pie_customization(fig_full, data, label_col, color_map, pull_categories, pull_amount, rotation_deg)
        show_chart(fig_full, n_categories=len(data), is_pie=True, unit_caption=unit_full, font_size=font_full)
    with pc_others:
        st.caption("🗂️ With 'Others'")
        title_others, unit_others, font_others = chart_controls(
            default_title=f"{title_prefix} - Others", auto_unit_caption=auto_unit_caption,
            key_prefix=f"{key_prefix}_others",
        )
        fig_others = px.pie(
            data_with_others, names=label_col, values=value_col, title=title_others,
            color=label_col, color_discrete_map=color_map,
        )
        fig_others.update_traces(
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
            hovertemplate="%{label}<br>Percentage: %{percent:.2%}<br>Value: %{value:,.2~f}<extra></extra>",
        )
        apply_pie_customization(
            fig_others, data_with_others, label_col, color_map, pull_categories, pull_amount, rotation_deg
        )
        show_chart(
            fig_others, n_categories=len(data_with_others), is_pie=True, unit_caption=unit_others,
            font_size=font_others,
        )


def render_renamable_table(df: pd.DataFrame, key_prefix: str, height: int = None) -> None:
    """Show an EDITABLE table: column headers can be custom-renamed by
    the user (only for THIS displayed table - it does not change the
    original column names in df_filtered/df_raw, so other filters/charts/
    calculations keep working normally against the original column
    names), AND the cell values themselves can be edited directly in the
    grid (useful for spot-fixing a value or trying out a quick what-if
    edit before exporting). Editing a cell here only changes what's shown
    in THIS table/export - it does not feed back into the filters, KPIs,
    or charts elsewhere in the dashboard."""
    with st.expander("✏️ Rename Columns (this table's display only)", expanded=False):
        st.caption(
            "Renaming a header here only affects the DISPLAY of this table - "
            "the original column name used by other filters/charts/calculations is unchanged."
        )
        rename_map = {}
        cols_list = list(df.columns)
        cols_per_row = 3
        for row_start in range(0, len(cols_list), cols_per_row):
            row_cols = cols_list[row_start:row_start + cols_per_row]
            widgets = st.columns(len(row_cols))
            for i, col in enumerate(row_cols):
                new_name = widgets[i].text_input(
                    f"'{col}'", value=str(col), key=f"{key_prefix}_rename_{col}",
                )
                rename_map[col] = new_name.strip() or str(col)
    display_df = df.rename(columns=rename_map)
    st.caption("✏️ Cells in this table are editable — double-click a cell to change its value.")
    editor_kwargs = dict(use_container_width=True, num_rows="fixed", key=f"{key_prefix}_editor")
    if height:
        st.data_editor(display_df, height=height, **editor_kwargs)
    else:
        st.data_editor(display_df, **editor_kwargs)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert a dataframe into Excel file bytes for download."""
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
    """Guess the header row using a heuristic: the row with the most
    filled-in cells among the first few rows is usually the header row
    (title/blank rows above it are usually only partially filled)."""
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
    """Columns that still have no name (the original header was truly
    empty) are given a clearer name than pandas' default 'Unnamed: N'."""
    df = df.copy()
    new_cols = []
    counter = 1
    for c in df.columns:
        if str(c).lower().startswith("unnamed"):
            new_cols.append(f"Unnamed Column {counter}")
            counter += 1
        else:
            new_cols.append(c)
    df.columns = new_cols
    return df


@st.cache_data(show_spinner=False)
def load_and_process(file_bytes: bytes, sheet_name, header_row: int = 0,
                      assume_dot_comma_as_thousands: bool = False):
    """Load Excel from bytes, clean it, and run the auto-detection engine."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=header_row)

    # -----------------------------------------------------------------
    # WIDE TIME-SERIES AUTO-UNPIVOT: if the sheet turns out to be in WIDE
    # format (1 column = 1 month/period, headers are literal dates/years -
    # common in trade/statistics reports like the Global Trade Tracker or
    # national statistics bureaus), it's automatically 'melted' into long
    # format (1 row = 1 point in time) so the whole dashboard's features
    # can use it fully (date/month/year filters, KPIs, charts). Generic -
    # works for any Excel sheet with a wide time-series pattern, not just
    # one specific file.
    #
    # IMPORTANT: this MUST run BEFORE clean_column_names(), because if the
    # original column headers are real datetime objects (common when
    # Excel stores dates as date cells, not text), once
    # clean_column_names() runs str() on the column names, that datetime
    # object turns into full text including the time (e.g.
    # '2000-01-01 00:00:00') and fails to be recognized as a period column
    # again. That's why detection & unpivoting happen here first, using
    # the ORIGINAL/raw column names.
    # -----------------------------------------------------------------
    wide_period_date_col = None
    granularity, period_cols = detect_wide_period_columns(df)
    if period_cols:
        unit_hint = detect_sheet_unit_hint(file_bytes, sheet_name, header_row)
        value_col_name = f"Value ({unit_hint})" if unit_hint else "Value"
        wide_period_date_col = "Date"
        df = melt_wide_period_columns(df, period_cols, wide_period_date_col, value_col_name, granularity)
        # Rows with an empty/'-' value (no transaction recorded for that
        # period) are dropped - this doesn't change ANY totals (NaN is
        # already ignored when summing/averaging), but keeps the dataset
        # much more compact & the dashboard responsive even when the
        # original sheet has hundreds of period columns.
        df[value_col_name] = clean_numeric_string(df[value_col_name], assume_dot_comma_as_thousands)
        df = df[df[value_col_name].notna()].reset_index(drop=True)

    df = clean_column_names(df)
    df = drop_empty_columns(df)
    df = rename_unnamed_columns(df)

    # IMPORTANT: before checking for duplicates, pure running-number
    # columns (e.g. "No.") are IGNORED from the comparison. Otherwise,
    # rows that are actually the exact same transaction (just with a
    # different serial number) would never be detected as duplicates,
    # and would get double-counted in every Total/Summary.
    serial_cols = detect_pure_serial_columns(df)
    dedup_subset = [c for c in df.columns if c not in serial_cols]
    df = df.drop_duplicates(subset=dedup_subset if dedup_subset else None)

    df = try_convert_numeric(df, assume_dot_comma_as_thousands)

    date_cols = detect_date_columns(df)
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # If there's a date column resulting from a wide-format unpivot, that
    # column BEST represents the actual observation time (not some other
    # metadata date column that might exist in the file, e.g.
    # 'First'/'Last' which might just indicate a data-availability range)
    # - so it's prioritized as date_cols[0] so that derived Year/Month
    # values automatically follow it.
    if wide_period_date_col and wide_period_date_col in date_cols:
        date_cols = [wide_period_date_col] + [c for c in date_cols if c != wide_period_date_col]

    year_cols = detect_year_columns(df)
    month_cols = detect_month_columns(df)

    # If the unpivot result has 'year' granularity (an annual sheet, where
    # the 'Date' column contains plain year numbers rather than full
    # dates), make sure that column is also prioritized as year_cols[0] -
    # so the Year Filter automatically follows the unpivot result column,
    # not some other column (e.g. 'First'/'Last' - Year) that's just
    # data-availability metadata.
    if wide_period_date_col and granularity == "year" and wide_period_date_col in df.columns:
        if wide_period_date_col not in year_cols:
            year_cols = [wide_period_date_col] + year_cols
        else:
            year_cols = [wide_period_date_col] + [c for c in year_cols if c != wide_period_date_col]

    # If there's only a date column (no explicit year/month), automatically
    # extract Year, Month, Month Name from the first date column.
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

    # Separate ID/reference number columns (e.g. "Contract Ref.No",
    # "Shipment No.") from numeric columns that are genuinely meaningful
    # to sum/average.
    id_like_cols = detect_id_like_columns(df, numeric_cols_raw)
    numeric_cols = [c for c in numeric_cols_raw if c not in id_like_cols]

    exclude_from_category = set(date_cols) | set(year_cols) | set(month_cols) | set(numeric_cols)
    category_cols = detect_category_columns(df, exclude=list(exclude_from_category))
    # id_like_cols is DELIBERATELY not merged into category_cols here, so
    # as not to overload the sidebar Category Filter with hundreds of
    # checkboxes (e.g. Contract Ref.No). id_like_cols is kept separate in
    # meta specifically for the breakdown dropdown in the Summary section.

    # all_text_cols: ALL text columns WITHOUT the CATEGORY_MAX_UNIQUE cap,
    # used for the single-select dropdowns in Category Analysis/
    # Distribution/Group Comparison/Group Values/Summary, so columns with
    # very many unique values (e.g. Vessel Name, Customer Name) STILL show
    # up in those options, even though they don't appear as checkboxes in
    # the sidebar Category Filter (which is deliberately capped at
    # CATEGORY_MAX_UNIQUE).
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
    "Upload any Excel file — the dashboard will automatically detect the "
    "data structure and build the relevant filters, KPIs, and visualizations."
)

with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("👈 Please upload an Excel file (.xlsx) in the sidebar to start the analysis.")
    st.stop()

file_bytes = uploaded_file.getvalue()

# Choose a sheet if the file has multiple sheets
try:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_names = xls.sheet_names
except Exception as e:
    st.error(f"Failed to read the Excel file: {e}")
    st.stop()

with st.sidebar:
    if len(sheet_names) > 1:
        selected_sheet = st.selectbox("Select Sheet", sheet_names, index=0)
    else:
        selected_sheet = sheet_names[0]

# -----------------------------------------------------------------
# HEADER ROW DETECTION: real-world files often have a title row or blank
# row(s) BEFORE the actual header row with column names. When this
# happens, columns can get misread (showing up as "Unnamed: N", etc) and
# all totals end up wrong. Show a raw preview + let the user correct the
# header row if the auto-guess is off.
# -----------------------------------------------------------------
guessed_header_row = guess_header_row(file_bytes, selected_sheet)

with st.expander("🧭 Check & Set Header Row (if you see 'Unnamed' columns / wrong totals)", expanded=False):
    st.caption(
        "Raw preview of the file's first 10 rows (before processing). "
        "Pick which row actually contains the real COLUMN NAMES."
    )
    raw_preview = pd.read_excel(
        io.BytesIO(file_bytes), sheet_name=selected_sheet, header=None, nrows=10
    )
    st.dataframe(raw_preview, use_container_width=True)
    # IMPORTANT: the key here has a sheet-name suffix (f"...{selected_sheet}")
    # so the widget RESETS every time the sheet changes. Without this
    # unique key, Streamlit treats it as the same widget and the default
    # "value=" no longer gets applied after the first interaction - so if
    # you switch sheets, the header-row number "sticks" to the previous
    # sheet's value, which can make the new sheet's columns read
    # incorrectly / appear to be missing.
    header_row = st.number_input(
        "Which row (starting from 0) is the header?",
        min_value=0, max_value=100, value=guessed_header_row, step=1,
        key=f"header_row_input_{selected_sheet}",
    )
    st.markdown("---")
    st.caption(
        "If a numeric column is TEXT with an ambiguous single dot/comma "
        "(e.g. \"8.847\"), it's treated by DEFAULT as a genuine decimal "
        "(8.847 stays 8.847) so that no number ever changes without you "
        "noticing. Only enable the option below if you're sure a single "
        "dot/comma is meant to be a thousands separator."
    )
    assume_dot_comma_as_thousands = st.checkbox(
        "Treat a SINGLE dot/comma in text numbers as a thousands separator "
        "(e.g. \"8.847\" -> 8847)",
        value=False,
    )

try:
    df_raw, meta = load_and_process(
        file_bytes, selected_sheet, header_row, assume_dot_comma_as_thousands
    )
except Exception as e:
    st.error(f"Failed to process the data: {e}")
    st.stop()

if df_raw.empty:
    st.warning("The Excel file was read successfully but there is no data to display.")
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
# HIDE COLUMNS: so columns that aren't relevant to the analysis (e.g.
# internal notes) can be removed from the entire dashboard without
# needing to edit the source Excel file. This is generic - it works for
# any column, nothing is hardcoded.
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    hidden_cols = st.multiselect(
        "🚫 Hide Columns (optional)",
        options=list(df_raw.columns),
        default=[],
        help="Columns selected here are removed from the entire dashboard: "
             "filters, KPIs, charts, and the data preview.",
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
# Debug panel: show the detection results as-is, so the user can quickly
# check if any column wasn't read the way they expected.
# -----------------------------------------------------------------
all_detected = set(date_cols) | set(year_cols) | set(month_cols) | set(numeric_cols) | set(all_text_cols)
undetected_cols = [c for c in df_raw.columns if c not in all_detected]

with st.expander("🔍 Columns Detected in This File (click to check)", expanded=False):
    if serial_cols:
        st.info(
            f"ℹ️ Pure running-number column(s) detected: **{', '.join(serial_cols)}** — "
            f"this column is IGNORED when checking for duplicate rows, so rows whose data "
            f"is otherwise identical (just with a different serial number) are still "
            f"counted as duplicates."
        )
    text_cols_over_limit = [c for c in all_text_cols if c not in category_cols]
    if text_cols_over_limit:
        st.info(
            f"ℹ️ Text column(s) with more than {CATEGORY_MAX_UNIQUE} unique values detected: "
            f"**{', '.join(text_cols_over_limit)}** — this column does NOT appear as a "
            f"checkbox in the sidebar Category Filter (so the sidebar doesn't get flooded), "
            f"but is STILL fully available in the Category Analysis, Distribution, "
            f"Group Comparison, Group Values, and Summary breakdown dropdowns."
        )
    d1, d2, d3 = st.columns(3)
    d1.markdown("**📅 Date Columns**")
    d1.write(date_cols if date_cols else "-")
    d1.markdown("**📆 Year Columns**")
    d1.write(year_cols if year_cols else "-")
    d1.markdown("**🗓️ Month Columns**")
    d1.write(month_cols if month_cols else "-")
    d2.markdown("**🔢 Numeric Columns** (summed/averaged)")
    d2.write(numeric_cols if numeric_cols else "-")
    d2.markdown("**🆔 ID / Reference Columns** (numeric but not meant to be totaled)")
    d2.write(id_like_cols if id_like_cols else "-")
    d3.markdown("**🏷️ Text/Category Columns** (all, including those with many unique values)")
    d3.write(all_text_cols if all_text_cols else "-")
    if undetected_cols:
        st.markdown("**⚠️ Not yet classified into any category** (likely mixed data types):")
        st.write(undetected_cols)

# -----------------------------------------------------------------
# EXCLUDE VALUES: the user can drop specific values from any column so
# they're excluded from the entire dashboard (KPIs, Summary, charts,
# Data Preview, everything). Different from "Group Values" (which
# MERGES) - this DROPS entirely. Generic, works for any column & value
# depending on the file's content.
# -----------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    with st.expander("🚫 Exclude Values from the Table (optional)", expanded=False):
        st.caption(
            "Pick specific values to be TOTALLY DROPPED from the entire dashboard "
            "(excluded from KPIs, Summary, charts, and any Data Preview)."
        )
        exclude_target_cols = st.multiselect(
            "Column(s) to set exclusions for",
            options=list(df_raw.columns),
            key="exclude_target_cols",
        )

        exclude_rules = {}
        for ecol in exclude_target_cols:
            ecol_options = sorted(df_raw[ecol].dropna().astype(str).unique().tolist())
            selected_exclude_vals = st.multiselect(
                f"Value(s) to exclude in '{ecol}'",
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
    st.sidebar.success(f"✅ {total_excluded_rows:,} row(s) excluded from the table.")

# -----------------------------------------------------------------
# GROUP VALUES (CUSTOM GROUPING): the user can merge several values from
# 1 category column into 1 new group (e.g. values "A", "B", "C" merged
# into a new group "ABC"), while values not selected stay as-is. The
# result can be saved as a SEPARATE NEW COLUMN, OR merged DIRECTLY into
# another existing column (e.g. the grouping result is considered
# "Indonesia" and placed into the "Country" column, alongside other
# countries that already exist there). Generic - works for any column &
# value depending on the uploaded file's content.
# -----------------------------------------------------------------
if "custom_groups" not in st.session_state:
    st.session_state.custom_groups = {}  # {source_col: [{"name":..., "values":[...], "target_col": str|None}]}


def get_merge_groups_for_target(target_col: str) -> list:
    """Return every group (from any source column) whose merge RESULT was
    applied onto a given `target_col`, e.g. the 'Jakarta' group (from the
    'City' column, original values abc/abcd/abcde) merged into the
    'Country' column. Used for the drill-down feature: as soon as the
    user selects a target_col that has a merged value, we can show a
    breakdown of its original contents."""
    result = []
    for src_col, groups in st.session_state.custom_groups.items():
        for g in groups:
            if g.get("target_col") == target_col:
                result.append({"name": g["name"], "values": g["values"], "src_col": src_col})
    return result

with st.sidebar:
    st.markdown("---")
    with st.expander("🗂️ Group Values (optional)", expanded=False):
        st.caption(
            "Merge several values into 1 new group as you need. The result can be saved "
            "as a separate new column, or merged directly into an existing column so its "
            "values sit alongside the other values already in that column."
        )
        group_source_col = st.selectbox(
            "SOURCE column (the one to be re-grouped)",
            options=["(Not used)"] + all_text_cols,
            key="group_source_col",
        )

        if group_source_col != "(Not used)":
            if group_source_col not in st.session_state.custom_groups:
                st.session_state.custom_groups[group_source_col] = []

            all_groups_this_source = st.session_state.custom_groups[group_source_col]

            if all_groups_this_source:
                st.markdown("**Groups created so far:**")
                for i, g in enumerate(all_groups_this_source):
                    target_desc = "→ new column" if not g.get("target_col") else f"→ merged into '{g['target_col']}'"
                    gcol1, gcol2 = st.columns([4, 1])
                    gcol1.write(f"🔸 **{g['name']}** {target_desc}: {', '.join(g['values'])}")
                    if gcol2.button("🗑️", key=f"del_group_{group_source_col}_{i}"):
                        st.session_state.custom_groups[group_source_col].pop(i)
                        st.rerun()

            st.markdown("**➕ Create a new group:**")

            other_cols_for_target = [c for c in all_text_cols if c != group_source_col]
            target_options = ["Save as a dedicated new column"] + [
                f"{c}  (merge into this column)" for c in other_cols_for_target
            ]
            target_choice = st.selectbox(
                "Where should this group's result go?",
                options=target_options,
                key="new_group_target_choice",
                help="'New column' creates a separate column just for this grouping. "
                     "Pick another column's name to overwrite the values in that column "
                     "(only for matching rows - other rows in that column stay as-is).",
            )
            target_col_value = (
                None if target_choice == "Save as a dedicated new column"
                else target_choice.replace("  (merge into this column)", "")
            )

            # A value is considered "already used" ONLY compared against
            # other groups with the EXACT SAME target (same target_col) -
            # so that 1 value (e.g. "Jakarta") can be used in the
            # "Indonesia" group (target: Country) AND ALSO reused to
            # create a sub-group inside Indonesia (target: another new
            # column).
            groups_same_target = [g for g in all_groups_this_source if g.get("target_col") == target_col_value]
            already_grouped_for_target = set()
            for g in groups_same_target:
                already_grouped_for_target.update(g["values"])

            all_source_values = sorted(df_raw[group_source_col].dropna().astype(str).unique().tolist())
            available_values = [v for v in all_source_values if v not in already_grouped_for_target]

            if available_values:
                new_group_name = st.text_input("New group name (e.g. 'Indonesia')", key="new_group_name_input")
                new_group_values = st.multiselect(
                    "Select the value(s) to merge into this group",
                    options=available_values,
                    key="new_group_values_input",
                )
                if st.button("➕ Add Group", key="add_group_btn"):
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
                        st.warning("Enter a group name and select at least 1 value first.")
            else:
                st.caption("All relevant values for this target are already in a group.")

# Apply grouping to df_raw. Two modes:
# 1) target_col empty -> create a NEW COLUMN "{src_col} (Grouped)"
# 2) target_col set    -> OVERWRITE values in that target column (only
#    for matching rows); other rows in the target column keep their
#    original values.
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
        new_col_name = f"{src_col} (Grouped)"
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
                    "Select Year(s)", options=year_values, default=year_values,
                    # The key is tied to the actual data (the year range) -
                    # so if the file changes (a different year range), the
                    # widget automatically resets to the full default,
                    # rather than sticking to the old file's selection,
                    # which could accidentally filter out every row.
                    key=f"year_filter_{year_col}_{len(year_values)}_{year_values[0]}_{year_values[-1]}",
                )
            if selected_years:
                df_filtered = df_filtered[df_filtered[year_col].isin(selected_years)]

    # -----------------------------------------------------------------
    # A. TIME FILTER - Month (robust: supports Indonesian/English month
    # names, abbreviations, numbers 1-12, AND columns that actually
    # contain full dates like '2024-01-01' - automatically extracting
    # just the month, bug-free).
    # -----------------------------------------------------------------
    if month_cols:
        month_col = month_cols[0]
        raw_month_values = df_raw[month_col].dropna().unique().tolist()

        def _to_month_name(v):
            num = month_to_number(v)
            if pd.isna(num):
                return None
            return MONTH_ORDER[int(num) - 1]

        # display_name: the standard month name (January, February, etc)
        # when the value is recognized (including abbreviations like
        # "Jan"/"Feb"/"Mar" in either English or Indonesian, or even a
        # full date that automatically gets its month extracted). If the
        # format isn't recognized, USE THE RAW VALUE AS-IS as the display
        # name - so EVERY value always shows up in the filter, nothing
        # gets silently skipped.
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
                    "Select Month(s)", options=options_display, default=options_display,
                    # The key is tied to the actual data (option count +
                    # first option) - so if the Excel file changes (the
                    # months that appear differ), the widget automatically
                    # resets to the full default, rather than sticking to
                    # the previous file's selection, which could
                    # accidentally filter out every row ("No data matches
                    # the current filter combination").
                    key=f"month_filter_{month_col}_{len(options_display)}_{options_display[0]}",
                )
            if selected_months_display:
                allowed_raw_vals = [v for v in raw_month_values if display_map[v] in selected_months_display]
                df_filtered = df_filtered[df_filtered[month_col].isin(allowed_raw_vals)]

    # -----------------------------------------------------------------
    # B. CATEGORY FILTER (checkbox) - DELIBERATELY still uses category_cols,
    # which is capped at CATEGORY_MAX_UNIQUE, so the sidebar doesn't get
    # flooded with hundreds/thousands of checkboxes for a column with many
    # unique values (e.g. Vessel Name). Such columns are still fully
    # usable in other dropdowns (Category Analysis, Distribution, etc)
    # via all_text_cols.
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
                    # The key is tied to this data's ACTUAL min/max - so if
                    # the user changes the Excel file (the number range
                    # differs), the slider automatically resets to the
                    # full default, rather than sticking to an old value
                    # that could be outside the new range (which would
                    # silently filter out every row).
                    key=f"num_filter_{col}_{round(min_v, 6)}_{round(max_v, 6)}",
                )
                df_filtered = df_filtered[
                    df_filtered[col].between(sel_range[0], sel_range[1]) | df_filtered[col].isna()
                ]

if df_filtered.empty:
    st.warning("No data matches the current combination of filters.")
    st.stop()

# =====================================================================
# PART 3B: LIVE PREVIEW FOR A NEWLY CREATED GROUP
# As soon as a new group is created in the sidebar ("Group Values"), its
# chart IMMEDIATELY appears here automatically - no need to switch tabs
# or click anything else. Change the "Value" or "Show as" option below,
# and the chart updates instantly (this is how Streamlit works: any
# widget change re-runs the page instantly) - so it feels "instant" even
# though it isn't real drag-and-drop.
# =====================================================================
active_grouped_cols = [
    f"{src_col} (Grouped)"
    for src_col, groups in st.session_state.custom_groups.items()
    if groups and f"{src_col} (Grouped)" in df_filtered.columns
]

if active_grouped_cols:
    st.markdown("## 🔍 Preview of Newly Created Group")
    st.caption(
        "Automatically shows up as soon as you create a new group in the sidebar. "
        "Change the settings below and the chart updates instantly."
    )
    for grouped_col in active_grouped_cols:
        with st.container(border=True):
            st.markdown(f"#### {grouped_col}")
            pv1, pv2 = st.columns(2)
            with pv1:
                preview_value_choice = st.selectbox(
                    "Value",
                    options=["Row Count"] + numeric_cols,
                    key=f"preview_value_{grouped_col}",
                )
            with pv2:
                preview_chart_type = st.selectbox(
                    "Show as",
                    options=["Bar Chart", "Pie Chart", "Horizontal Bar"],
                    key=f"preview_chart_type_{grouped_col}",
                )

            if preview_value_choice == "Row Count":
                preview_data = df_filtered[grouped_col].value_counts().reset_index()
                preview_data.columns = [grouped_col, "Value"]
                preview_label = "Row Count"
                preview_unit_source = None
            else:
                preview_data = df_filtered.groupby(grouped_col, as_index=False)[preview_value_choice].sum()
                preview_data.columns = [grouped_col, "Value"]
                preview_label = f"Total {preview_value_choice}"
                preview_unit_source = preview_value_choice
            preview_data = preview_data.sort_values("Value", ascending=False)
            preview_data_scaled, preview_auto_unit_caption = scale_for_display(
                preview_data, "Value", preview_unit_source
            )
            preview_default_title = f"{preview_label} per {grouped_col}"
            preview_title, preview_unit_caption, preview_font_size = chart_controls(
                default_title=preview_default_title, auto_unit_caption=preview_auto_unit_caption,
                key_prefix=f"preview_{grouped_col}",
            )

            if preview_chart_type == "Bar Chart":
                fig_preview = px.bar(
                    preview_data_scaled, x=grouped_col, y="Value", color=grouped_col,
                    text="Value", title=preview_title,
                )
                fig_preview.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")
            elif preview_chart_type == "Pie Chart":
                preview_color_map, preview_pull_cats, preview_pull_amt, preview_rotation = pick_pie_colors_and_pull(
                    preview_data_scaled[grouped_col].astype(str).tolist(), section_key=f"preview_{grouped_col}"
                )
                fig_preview = px.pie(
                    preview_data_scaled, names=grouped_col, values="Value", color=grouped_col,
                    color_discrete_map=preview_color_map,
                    title=preview_title,
                )
                fig_preview.update_traces(
                    texttemplate="%{label}<br>%{percent:.2%}<br>%{value:,.2~f}",
                    hovertemplate="%{label}<br>Percentage: %{percent:.2%}<br>Value: %{value:,.2~f}<extra></extra>",
                )
                apply_pie_customization(
                    fig_preview, preview_data_scaled, grouped_col, preview_color_map,
                    preview_pull_cats, preview_pull_amt, preview_rotation,
                )
            else:
                preview_data_h = preview_data_scaled.sort_values("Value")
                fig_preview = px.bar(
                    preview_data_h, x="Value", y=grouped_col, color=grouped_col, orientation="h",
                    text="Value", title=preview_title,
                )
                fig_preview.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")

            show_chart(
                fig_preview, n_categories=len(preview_data), is_pie=(preview_chart_type == "Pie Chart"),
                unit_caption=preview_unit_caption, font_size=preview_font_size,
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
    st.caption("The Total & Average below are calculated from ALL data that passes the current "
               "filters (not split out per category/company).")
    # Show every numeric column, split into rows of 4 columns (Total + Avg)
    # so a column like Volume still shows its total even if it isn't
    # among the first 4 columns.
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
# PART 4B: SUMMARY (flexible, the user chooses what to see)
# =====================================================================

st.markdown("## 📊 Summary")
st.caption(
    "Pick your own numeric column(s), aggregation method, and what to break it down by. "
    "The Grand Total below is always calculated from all data passing the current filters."
)

if numeric_cols:
    # Breakdown options: ALL text columns (all_text_cols, no unique-value
    # cap) COMBINED with ID/reference number columns (Contract Ref.No,
    # Shipment No., etc) which, despite being numeric dtype, are
    # semantically an identifier -> still need to be usable for breakdown/
    # totaling per group, including columns with many unique values
    # (Vessel Name, etc).
    breakdown_options = all_text_cols + [c for c in id_like_cols if c not in all_text_cols]

    s1, s2, s3 = st.columns([2, 1.3, 2])

    with s1:
        summary_value_cols = st.multiselect(
            "Numeric column(s) to summarize",
            options=numeric_cols,
            default=numeric_cols[:1],
            key="summary_value_cols",
        )
    with s2:
        agg_label_map = {
            "Total (Sum)": "sum",
            "Average (Mean)": "mean",
            "Row Count": "count",
            "Minimum": "min",
            "Maximum": "max",
        }
        agg_label = st.selectbox(
            "Aggregation Type", options=list(agg_label_map.keys()), key="summary_agg"
        )
        agg_func = agg_label_map[agg_label]
    with s3:
        breakdown_col = st.selectbox(
            "Break down by (optional)",
            options=["(No breakdown / Grand Total only)"] + breakdown_options,
            key="summary_breakdown",
            help="You can pick any category column (Company, Country, Vessel Name, etc) OR "
                 "an ID/reference column like Contract Ref.No — useful when a single value "
                 "of that column appears across many rows.",
        )

    if summary_value_cols:
        # ------------- GRAND TOTAL (always from all filtered data) -------------
        st.markdown("#### Grand Total")
        grand_cols = st.columns(len(summary_value_cols))
        for i, col in enumerate(summary_value_cols):
            grand_val = df_filtered[col].count() if agg_func == "count" \
                else getattr(df_filtered[col], agg_func)()
            grand_cols[i].metric(f"{agg_label} - {col}", fmt_num(grand_val))

        # ------------- BREAKDOWN TABLE (if the user picked a breakdown column) -------------
        if breakdown_col != "(No breakdown / Grand Total only)":
            st.markdown(f"#### Breakdown by {breakdown_col}")
            n_unique_breakdown = df_filtered[breakdown_col].nunique(dropna=True)
            st.caption(f"{n_unique_breakdown:,} unique value(s) found in this column.")

            summary_table = (
                df_filtered.groupby(breakdown_col)[summary_value_cols]
                .agg(agg_func)
                .reset_index()
            )
            # Add Row Count per group - if the number is higher than
            # expected (e.g. 1 Contract Ref.No should only appear on 2-3
            # rows but shows up on 15), that's the first clue there's a
            # duplicate/repeated entry.
            row_counts = df_filtered.groupby(breakdown_col).size().reset_index(name="Row Count")
            summary_table = summary_table.merge(row_counts, on=breakdown_col)
            summary_table = summary_table.sort_values(summary_value_cols[0], ascending=False)

            render_renamable_table(summary_table, key_prefix=f"summarytable_{breakdown_col}", height=400)

            summary_excel_bytes = to_excel_bytes(summary_table)
            st.download_button(
                label="⬇️ Download This Summary (Excel)",
                data=summary_excel_bytes,
                file_name=f"summary_by_{breakdown_col}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_summary",
            )

            # ------------- DRILL-DOWN: check the raw rows behind 1 specific value -------------
            st.markdown("#### 🔎 Check Raw Row Detail")
            st.caption(
                "If a total looks off (e.g. very different from your own pivot table), "
                "select 1 value below to see ALL the raw rows that make up that number — "
                "usually you'll spot a duplicate row right away."
            )
            drilldown_options = sorted(df_filtered[breakdown_col].dropna().astype(str).unique().tolist())
            drilldown_value = st.selectbox(
                f"Select a {breakdown_col} value to inspect",
                options=["(Select one)"] + drilldown_options,
                key="drilldown_value",
            )
            if drilldown_value != "(Select one)":
                drilldown_rows = df_filtered[df_filtered[breakdown_col].astype(str) == drilldown_value]
                st.write(
                    f"Found **{len(drilldown_rows):,} raw row(s)** for "
                    f"`{breakdown_col} = {drilldown_value}`."
                )
                for col in summary_value_cols:
                    st.write(
                        f"- Total **{col}** from these {len(drilldown_rows):,} row(s): "
                        f"**{fmt_num(drilldown_rows[col].sum())}**"
                    )
                render_renamable_table(drilldown_rows, key_prefix=f"drilldownrows_{breakdown_col}_{drilldown_value}", height=350)
    else:
        st.info("Select at least 1 numeric column above to see the summary.")
else:
    st.info("No numeric column detected to build a summary from.")

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
            cat_col = st.selectbox("Select Category", all_text_cols, key="cat_col_bar")
        with cc2:
            val_col = st.selectbox("Select Numeric", numeric_cols, key="cat_val_bar")
        with cc3:
            top_n = st.selectbox("Show Top", options=[1, 3, 5, 10, 15, 20], index=3, key="top_n_bar")

        cat_agg = df_filtered.groupby(cat_col, as_index=False)[val_col].sum()
        cat_agg = cat_agg.sort_values(val_col, ascending=False)
        cat_agg_scaled, cat_auto_unit_caption = scale_for_display(cat_agg, val_col, val_col)

        b1, b2 = st.columns(2)
        with b1:
            default_title_left = f"{cat_col} vs Total {val_col}"
            title_left, unit_left, font_left = chart_controls(
                default_title=default_title_left, auto_unit_caption=cat_auto_unit_caption,
                key_prefix=f"catbar_left_{cat_col}_{val_col}",
            )
            fig_bar = px.bar(
                cat_agg_scaled, x=cat_col, y=val_col,
                title=title_left,
            )
            show_chart(fig_bar, n_categories=len(cat_agg), unit_caption=unit_left, font_size=font_left)
        with b2:
            top_n_data = cat_agg.head(top_n).sort_values(val_col)
            top_n_data_scaled, top_n_auto_unit_caption = scale_for_display(top_n_data, val_col, val_col)
            default_title_right = f"Top {top_n} {cat_col} by {val_col}"
            title_right, unit_right, font_right = chart_controls(
                default_title=default_title_right, auto_unit_caption=top_n_auto_unit_caption,
                key_prefix=f"catbar_right_{cat_col}_{val_col}_{top_n}",
            )
            fig_hbar = px.bar(
                top_n_data_scaled, x=val_col, y=cat_col, orientation="h",
                title=title_right,
                text=val_col,
            )
            fig_hbar.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")
            show_chart(fig_hbar, n_categories=len(top_n_data), unit_caption=unit_right, font_size=font_right)
    else:
        st.info("No matching category and numeric column combination found for Category Analysis.")

# -----------------------------------------------------------------
# B. DISTRIBUTION
# -----------------------------------------------------------------
with tabs[1]:
    if all_text_cols:
        dc1, dc2 = st.columns(2)
        with dc1:
            dist_col = st.selectbox("Select Category for Distribution", all_text_cols, key="dist_col")
        with dc2:
            dist_value_options = ["Row Count"] + numeric_cols
            dist_value_choice = st.selectbox(
                "Based On", options=dist_value_options, key="dist_value_choice"
            )

        if dist_value_choice == "Row Count":
            dist_data = df_filtered[dist_col].value_counts().reset_index()
            dist_data.columns = [dist_col, "Value"]
            value_label = "Row Count"
            dist_unit_source = None
        else:
            dist_data = df_filtered.groupby(dist_col, as_index=False)[dist_value_choice].sum()
            dist_data.columns = [dist_col, "Value"]
            value_label = f"Total {dist_value_choice}"
            dist_unit_source = dist_value_choice

        dist_data = dist_data.sort_values("Value", ascending=False)

        others_selected_dist = st.multiselect(
            "Combine the following categories into 'Others' (optional)",
            options=dist_data[dist_col].astype(str).tolist(),
            key="others_selected_dist",
            help="Select which categories should be merged into a single 'Others' slice on the "
                 "right-hand pie. The left-hand pie always stays full, without 'Others'.",
        )
        dist_color_map, dist_pull_cats, dist_pull_amt, dist_rotation = pick_pie_colors_and_pull(
            dist_data[dist_col].astype(str).tolist(), section_key=f"dist_{dist_col}"
        )
        render_full_vs_others_pies(
            dist_data, dist_col, "Value", f"Distribution of {dist_col} ({value_label})", others_selected_dist,
            unit_source_col=dist_unit_source,
            color_map=dist_color_map, pull_categories=dist_pull_cats,
            pull_amount=dist_pull_amt, rotation_deg=dist_rotation,
            key_prefix=f"dist_{dist_col}",
        )

        with st.expander("📋 View detailed numbers per category"):
            dist_data_display = dist_data.copy()
            total_val = dist_data_display["Value"].sum()
            dist_data_display["Percentage"] = (dist_data_display["Value"] / total_val * 100).map(lambda x: f"{x:.2f}%")
            dist_data_display["Value"] = dist_data_display["Value"].map(fmt_num)
            dist_data_display.columns = [dist_col, value_label, "Percentage"]
            render_renamable_table(dist_data_display, key_prefix=f"distdetail_{dist_col}")

        # -----------------------------------------------------------------
        # DRILL-DOWN: if the selected column (dist_col) has a value that
        # came from "Group Values -> merged into this column" (e.g. the
        # 'City' column has a value 'Jakarta' that's actually a merge of
        # 'abc', 'abcd', 'abcde'), show a second pie chart that breaks down
        # that group's original contents - exactly like the 'Jakarta' pie
        # splitting into abc/abcd/abcde next to the 'City' pie in the
        # original sketch.
        # -----------------------------------------------------------------
        merge_groups_for_dist = get_merge_groups_for_target(dist_col)
        if merge_groups_for_dist:
            st.markdown("---")
            st.markdown("#### 🔍 Group Detail (Drill-Down)")
            st.caption(
                f"Column **{dist_col}** has value(s) that came from merging via the "
                "'Group Values' feature. Pick a group below to see a breakdown of its "
                "original contents (e.g. 'Jakarta' broken back down into abc, abcd, abcde)."
            )
            drill_group_names = [g["name"] for g in merge_groups_for_dist]
            selected_drill_group = st.selectbox(
                "Select a group to break down", options=drill_group_names, key="drill_group_select"
            )
            drill_info = next(g for g in merge_groups_for_dist if g["name"] == selected_drill_group)
            drill_src_col = drill_info["src_col"]

            df_drill = df_filtered[df_filtered[dist_col].astype(str) == selected_drill_group]

            if drill_src_col in df_drill.columns and not df_drill.empty:
                if dist_value_choice == "Row Count":
                    drill_data = df_drill[drill_src_col].value_counts().reset_index()
                    drill_data.columns = [drill_src_col, "Value"]
                else:
                    drill_data = df_drill.groupby(drill_src_col, as_index=False)[dist_value_choice].sum()
                    drill_data.columns = [drill_src_col, "Value"]
                drill_data = drill_data.sort_values("Value", ascending=False)
                others_selected_drill = st.multiselect(
                    "Combine the following categories into 'Others' (optional)",
                    options=drill_data[drill_src_col].astype(str).tolist(),
                    key="others_selected_drill",
                    help="Select which categories should be merged into a single 'Others' slice on the right-hand pie.",
                )
                drill_color_map, drill_pull_cats, drill_pull_amt, drill_rotation = pick_pie_colors_and_pull(
                    drill_data[drill_src_col].astype(str).tolist(),
                    section_key=f"drill_{selected_drill_group}_{drill_src_col}",
                )
                render_full_vs_others_pies(
                    drill_data, drill_src_col, "Value",
                    f"Breakdown of '{selected_drill_group}' by {drill_src_col} ({value_label})",
                    others_selected_drill,
                    unit_source_col=dist_unit_source,
                    color_map=drill_color_map, pull_categories=drill_pull_cats,
                    pull_amount=drill_pull_amt, rotation_deg=drill_rotation,
                    key_prefix=f"drill_{selected_drill_group}_{drill_src_col}",
                )

                with st.expander(f"📋 View detailed numbers for '{selected_drill_group}'"):
                    drill_data_display = drill_data.copy()
                    total_drill_val = drill_data_display["Value"].sum()
                    drill_data_display["Percentage"] = (
                        drill_data_display["Value"] / total_drill_val * 100
                    ).map(lambda x: f"{x:.2f}%")
                    drill_data_display["Value"] = drill_data_display["Value"].map(fmt_num)
                    drill_data_display.columns = [drill_src_col, value_label, "Percentage"]
                    render_renamable_table(drill_data_display, key_prefix=f"drilldetail_{selected_drill_group}_{drill_src_col}")
            else:
                st.info(f"No data found for the group '{selected_drill_group}' under the current filters/column.")
    else:
        st.info("No category column found to build a distribution chart.")

# -----------------------------------------------------------------
# C. NUMERIC ANALYSIS
# -----------------------------------------------------------------
with tabs[2]:
    if numeric_cols:
        num_col = st.selectbox("Select Numeric Column", numeric_cols, key="num_analysis_col")
        num_auto_unit = detect_column_unit(num_col)
        num_auto_unit_caption = f"In {num_auto_unit}" if num_auto_unit else ""
        n1, n2 = st.columns(2)
        with n1:
            title_hist, unit_hist, font_hist = chart_controls(
                default_title=f"Histogram of {num_col}", auto_unit_caption=num_auto_unit_caption,
                key_prefix=f"hist_{num_col}",
            )
            fig_hist = px.histogram(df_filtered, x=num_col, title=title_hist)
            show_chart(fig_hist, unit_caption=unit_hist, font_size=font_hist)
        with n2:
            title_box, unit_box, font_box = chart_controls(
                default_title=f"Box Plot of {num_col}", auto_unit_caption=num_auto_unit_caption,
                key_prefix=f"box_{num_col}",
            )
            fig_box = px.box(df_filtered, y=num_col, title=title_box)
            show_chart(fig_box, unit_caption=unit_box, font_size=font_box)
    else:
        st.info("No numeric column found to analyze the numeric distribution.")

# -----------------------------------------------------------------
# D. RELATIONSHIP
# -----------------------------------------------------------------
with tabs[3]:
    if len(numeric_cols) >= 2:
        r1, r2 = st.columns(2)
        with r1:
            x_col = st.selectbox("X Axis", numeric_cols, index=0, key="scatter_x")
        with r2:
            y_default_idx = 1 if len(numeric_cols) > 1 else 0
            y_col = st.selectbox("Y Axis", numeric_cols, index=y_default_idx, key="scatter_y")

        color_arg = all_text_cols[0] if all_text_cols else None
        title_scatter, unit_scatter, font_scatter = chart_controls(
            default_title=f"{x_col} vs {y_col}", auto_unit_caption="",
            key_prefix=f"scatter_{x_col}_{y_col}",
        )
        fig_scatter = px.scatter(
            df_filtered, x=x_col, y=y_col, color=color_arg,
            title=title_scatter,
        )
        show_chart(fig_scatter, unit_caption=unit_scatter, font_size=font_scatter)
    else:
        st.info("At least 2 numeric columns are needed to build a Scatter Plot.")

# -----------------------------------------------------------------
# E. GROUP COMPARISON (generic - works for any column & value,
# not tied to a specific business term/case)
# -----------------------------------------------------------------
with tabs[4]:
    st.caption(
        "Split the data into 2 groups based on a column & values of your choosing "
        "(e.g. Domestic vs Export, Active vs Inactive, or anything based on the file's "
        "content), then break down one of the groups further per sub-category with "
        "different colors."
    )
    if all_text_cols and numeric_cols:
        gc1, gc2 = st.columns(2)
        with gc1:
            group_class_col = st.selectbox(
                "Classification Column (to split into 2 groups)",
                options=all_text_cols,
                key="group_class_col",
                help="The category column used to split the data into 2 groups, e.g. a "
                     "Region, Type, or Status column.",
            )
        with gc2:
            group_value_col = st.selectbox(
                "Numeric Column to Analyze",
                options=numeric_cols,
                key="group_value_col",
            )
        show_unit_note(group_value_col)

        group_class_options = sorted(df_filtered[group_class_col].dropna().astype(str).unique().tolist())
        group_a_name = st.text_input(
            "Name for Group A (free text, display label only)",
            value="Group A", key="group_a_label",
        )
        selected_group_a_values = st.multiselect(
            f"Select the value(s) that belong to '{group_a_name}' (the rest automatically become the other group)",
            options=group_class_options,
            key="group_a_selected_values",
        )

        other_category_cols = [c for c in all_text_cols if c != group_class_col]
        group_sub_col = None
        if other_category_cols:
            group_sub_col = st.selectbox(
                "Sub-Category Column (optional, for a more detailed breakdown)",
                options=["(No further breakdown)"] + other_category_cols,
                key="group_sub_col",
            )
            if group_sub_col == "(No further breakdown)":
                group_sub_col = None

        if selected_group_a_values:
            df_group_a = df_filtered[df_filtered[group_class_col].astype(str).isin(selected_group_a_values)]
            df_group_b = df_filtered[~df_filtered[group_class_col].astype(str).isin(selected_group_a_values)]
            group_b_name = f"Not {group_a_name}"

            st.markdown(f"#### Comparison: {group_a_name} vs {group_b_name}")
            comp1, comp2 = st.columns(2)
            group_a_total = df_group_a[group_value_col].sum()
            group_b_total = df_group_b[group_value_col].sum()
            grand_total_group = group_a_total + group_b_total
            comp1.metric(
                f"Total {group_value_col} - {group_a_name}",
                fmt_num(group_a_total),
                f"{(group_a_total/grand_total_group*100 if grand_total_group else 0):.2f}% of total",
            )
            comp2.metric(
                f"Total {group_value_col} - {group_b_name}",
                fmt_num(group_b_total),
                f"{(group_b_total/grand_total_group*100 if grand_total_group else 0):.2f}% of total",
            )

            if group_sub_col:
                st.markdown(f"#### {group_a_name} Breakdown by {group_sub_col}")
                group_breakdown = (
                    df_group_a.groupby(group_sub_col, as_index=False)[group_value_col]
                    .sum()
                    .sort_values(group_value_col, ascending=False)
                )
                group_row_counts = df_group_a.groupby(group_sub_col).size().reset_index(name="Row Count")
                group_breakdown = group_breakdown.merge(group_row_counts, on=group_sub_col)

                others_selected_comparison = st.multiselect(
                    "Combine the following categories into 'Others' in the pie chart (optional)",
                    options=group_breakdown[group_sub_col].astype(str).tolist(),
                    key="others_selected_comparison",
                    help="Select which categories should be merged into a single 'Others' slice.",
                )

                gcomp_color_map, gcomp_pull_cats, gcomp_pull_amt, gcomp_rotation = pick_pie_colors_and_pull(
                    group_breakdown[group_sub_col].astype(str).tolist(),
                    section_key=f"gcomp_{group_a_name}_{group_sub_col}",
                )
                render_full_vs_others_pies(
                    group_breakdown, group_sub_col, group_value_col,
                    f"{group_a_name} Distribution by {group_sub_col}",
                    others_selected_comparison,
                    unit_source_col=group_value_col,
                    color_map=gcomp_color_map, pull_categories=gcomp_pull_cats,
                    pull_amount=gcomp_pull_amt, rotation_deg=gcomp_rotation,
                    key_prefix=f"gcomp_{group_a_name}_{group_sub_col}",
                )

                group_breakdown_scaled, group_bar_auto_unit_caption = scale_for_display(
                    group_breakdown, group_value_col, group_value_col
                )
                title_gbar, unit_gbar, font_gbar = chart_controls(
                    default_title=f"Total {group_value_col} by {group_sub_col}",
                    auto_unit_caption=group_bar_auto_unit_caption,
                    key_prefix=f"gcompbar_{group_a_name}_{group_sub_col}",
                )
                fig_group_bar = px.bar(
                    group_breakdown_scaled, x=group_sub_col, y=group_value_col,
                    color=group_sub_col,
                    text=group_value_col,
                    title=title_gbar,
                )
                fig_group_bar.update_traces(texttemplate="%{text:,.2~f}", textposition="outside")
                show_chart(
                    fig_group_bar, n_categories=len(group_breakdown), unit_caption=unit_gbar, font_size=font_gbar
                )

                st.markdown("##### 📋 Detail Data")
                group_breakdown_display = group_breakdown.copy()
                total_group_val = group_breakdown_display[group_value_col].sum()
                group_breakdown_display["Percentage"] = (
                    group_breakdown_display[group_value_col] / total_group_val * 100
                ).map(lambda x: f"{x:.2f}%")
                group_breakdown_display[group_value_col] = group_breakdown_display[group_value_col].map(fmt_num)
                render_renamable_table(group_breakdown_display, key_prefix=f"groupbreakdown_{group_a_name}_{group_sub_col}")
            else:
                st.info("Select a Sub-Category column above to see a colorful breakdown per group.")
        else:
            st.info(f"Select at least 1 value that belongs to '{group_a_name}' above.")
    else:
        st.info("At least 1 category column and 1 numeric column are needed for Group Comparison.")

st.markdown("---")


# =====================================================================
# PART 6: DATA PREVIEW & EXPORT
# =====================================================================

st.markdown("## 🧾 Data Preview")
st.caption(f"Showing {len(df_filtered):,} row(s) after filtering (out of {len(df_raw):,} total rows).")
render_renamable_table(df_filtered, key_prefix="datapreview", height=380)


excel_bytes = to_excel_bytes(df_filtered)

st.download_button(
    label="⬇️ Download Filtered Excel",
    data=excel_bytes,
    file_name="filtered_dashboard_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")
st.caption(
    "This dashboard is generic — the filter structure, KPIs, and charts automatically "
    "adapt to whatever columns are detected in the uploaded Excel file."
)
