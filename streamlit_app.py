"""
Stock Screener Portal - Streamlit dashboard
Live data via yfinance. Combined fundamental + technical scoring.
Deployed on Streamlit Community Cloud.
"""

import math
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="סורק המניות", page_icon="📈", layout="wide")

# RTL support for Hebrew UI
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;600;800&display=swap');

    html, body, .stApp, [class*="css"] {
        font-family: 'Heebo', sans-serif !important;
    }
    .stApp { direction: rtl; }
    div[data-testid="stMarkdownContainer"] { text-align: right; }
    h1, h2, h3 { text-align: right; }

    /* Hero title */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2DD4A7, #4FA3F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #8B96A8;
        font-size: 0.95rem;
        margin-top: 0.2rem;
        margin-bottom: 1.2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        direction: rtl;
        text-align: right;
        background: linear-gradient(145deg, #161E2E, #1B2537);
        border: 1px solid #26314A;
        border-radius: 14px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label { color: #8B96A8 !important; }

    /* Expander */
    div[data-testid="stExpander"] {
        border: 1px solid #26314A;
        border-radius: 14px;
        background: #131A28;
    }

    /* Dataframe container */
    div[data-testid="stDataFrame"] {
        border: 1px solid #26314A;
        border-radius: 14px;
        overflow: hidden;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid #2DD4A7;
        color: #2DD4A7;
        background: transparent;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #2DD4A71A;
        border-color: #2DD4A7;
        color: #2DD4A7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

US_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    "JPM", "V", "JNJ", "PG", "XOM", "COST", "AMD", "INTC", "PFE",
]
TA_UNIVERSE = [
    "TEVA.TA", "LUMI.TA", "POLI.TA", "NICE.TA", "ESLT.TA",
    "ICL.TA", "DSCT.TA", "MZTF.TA", "PHOE.TA", "TSEM.TA",
]


# ----------------------------------------------------------------
# Scoring logic (same engine as phase 1)
# ----------------------------------------------------------------
def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def band_score(value, bands):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    score = bands[-1][1]
    for threshold, s in bands:
        if value <= threshold:
            score = s
            break
    return score


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, float("nan"))
    return (100 - 100 / (1 + rs)).iloc[-1]


def fundamental_score(info):
    parts = []
    pe = info.get("trailingPE")
    s = band_score(pe, [(10, 90), (18, 75), (28, 55), (40, 35), (1e9, 15)])
    if s is not None:
        parts.append((s, 0.25))
    growth = info.get("revenueGrowth")
    if growth is not None:
        parts.append((clamp(50 + growth * 100 * 2.5), 0.25))
    roe = info.get("returnOnEquity")
    if roe is not None:
        parts.append((clamp(roe * 100 * 4), 0.2))
    dte = info.get("debtToEquity")
    s = band_score(dte, [(30, 95), (60, 80), (100, 60), (200, 35), (1e9, 10)])
    if s is not None:
        parts.append((s, 0.15))
    fcf, mcap = info.get("freeCashflow"), info.get("marketCap")
    if fcf and mcap:
        parts.append((clamp(fcf / mcap * 100 * 12), 0.15))
    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


def technical_score(hist):
    if hist is None or len(hist) < 210:
        return None
    close = hist["Close"]
    price = close.iloc[-1]
    parts = []

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    ma_score = (30 if price > sma50 else 0) + (40 if price > sma200 else 0) \
        + (30 if sma50 > sma200 else 0)
    parts.append((ma_score, 0.35))

    r = rsi(close)
    if not math.isnan(r):
        if 45 <= r <= 65:
            s = 90
        elif 35 <= r < 45 or 65 < r <= 75:
            s = 60
        else:
            s = 25
        parts.append((s, 0.2))

    if len(close) > 63:
        mom3 = (price / close.iloc[-63] - 1) * 100
        parts.append((clamp(50 + mom3 * 2), 0.2))
    if len(close) > 126:
        mom6 = (price / close.iloc[-126] - 1) * 100
        parts.append((clamp(50 + mom6 * 1.2), 0.15))

    vol = hist["Volume"]
    v20 = vol.rolling(20).mean().iloc[-1]
    v90 = vol.rolling(90).mean().iloc[-1]
    if v90 and v90 > 0:
        parts.append((clamp(50 + (v20 / v90 - 1) * 100), 0.1))

    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w


# ----------------------------------------------------------------
# Data fetching (cached for 1 hour)
# ----------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_scores(universe):
    rows = []
    progress = st.progress(0, text="מושך נתונים...")
    for i, symbol in enumerate(universe):
        try:
            t = yf.Ticker(symbol)
            info = t.info
            hist = t.history(period="1y")
            f = fundamental_score(info)
            tech = technical_score(hist)
            if f is None and tech is None:
                continue
            rows.append({
                "סימול": symbol,
                "שם": (info.get("shortName") or "")[:28],
                "שוק": "ת\"א" if symbol.endswith(".TA") else "ארה\"ב",
                "פונדמנטלי": round(f, 1) if f is not None else None,
                "טכני": round(tech, 1) if tech is not None else None,
                "מחיר": round(hist["Close"].iloc[-1], 2) if len(hist) else None,
            })
        except Exception:
            continue
        progress.progress((i + 1) / len(universe), text=f"מושך נתונים... {symbol}")
    progress.empty()
    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# UI
# ----------------------------------------------------------------
st.markdown('<div class="hero-title">📈 סורק המניות</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">דירוג משולב פונדמנטלי + טכני · ארה"ב ות"א · '
    'מועמדות למחקר, לא המלצות השקעה</div>',
    unsafe_allow_html=True,
)

with st.expander("⚙️ הגדרות", expanded=False):
    market = st.radio("שוק", ["הכל", "ארה\"ב", "ת\"א"], horizontal=True)
    w_fund = st.slider("משקל פונדמנטלי (%)", 0, 100, 50, step=5)
    st.caption(f"משקל טכני: {100 - w_fund}%")
    min_score = st.slider("ציון כולל מינימלי", 0, 100, 0, step=5)
    if st.button("🔄 רענן נתונים"):
        st.cache_data.clear()
        st.rerun()

universe = US_UNIVERSE + TA_UNIVERSE
df = fetch_scores(universe)

if df.empty:
    st.error("לא התקבלו נתונים. נסה לרענן בעוד רגע.")
    st.stop()

# Compute weighted total with current slider values
w = w_fund / 100
df = df.copy()
df["ציון כולל"] = (
    df["פונדמנטלי"].fillna(50) * w + df["טכני"].fillna(50) * (1 - w)
).round(1)

if market != "הכל":
    df = df[df["שוק"] == market]
df = df[df["ציון כולל"] >= min_score]
df = df.sort_values("ציון כולל", ascending=False).reset_index(drop=True)
df.index = df.index + 1

col1, col2, col3 = st.columns(3)
col1.metric("מניות בסריקה", len(df))
if len(df):
    col2.metric("מובילה", df.iloc[0]["סימול"], f'{df.iloc[0]["ציון כולל"]}')
    col3.metric("ציון ממוצע", round(df["ציון כולל"].mean(), 1))

st.dataframe(
    df.style.background_gradient(
        subset=["פונדמנטלי", "טכני", "ציון כולל"], cmap="RdYlGn", vmin=0, vmax=100
    ),
    use_container_width=True,
    height=640,
)

if len(df):
    st.subheader("עשירייה מובילה")
    top = df.head(10).set_index("סימול")["ציון כולל"]
    st.bar_chart(top, color="#2DD4A7")
    
