from typing import Dict, List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
import io, math, os, logging, re, unicodedata, hashlib, json, datetime
from difflib import get_close_matches
import uvicorn
from sqlalchemy import create_engine, Column, Integer, Float, Date, MetaData, Table

# ----------------------- Setup -----------------------
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files")
BUNDLES_DIR = os.path.join(UPLOAD_DIR, "bundles")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(BUNDLES_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Betting Performance Analyzer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Database (kept for completeness) --------------------
DATABASE_URL = "sqlite:///betting_analysis.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
daily_stats_table = Table(
    'daily_stats', metadata,
    Column('Date_tip', Date, primary_key=True),
    Column('Tips', Integer),
    Column('Winners', Integer),
    Column('Profit', Float),
    Column('BSP_Mean', Float),
    Column('MorningWap_Mean', Float),
    Column('Units_Staked', Integer),
    Column('Units_Returned', Float),
    Column('ROI_percent', Float),
    Column('Win_Rate', Float),
    Column('CLV', Float),
    Column('Drifters_percent', Float),
    Column('Steamers_percent', Float),
)
metadata.create_all(engine)

# ----------------------- Helpers: saving & caching -----------------------
def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def _file_dir(file_hash: str) -> str:
    d = os.path.join(UPLOAD_DIR, file_hash[:2], file_hash[2:4], file_hash)
    os.makedirs(d, exist_ok=True)
    return d

def _bundle_dir(bundle_hash: str) -> str:
    d = os.path.join(BUNDLES_DIR, bundle_hash)
    os.makedirs(d, exist_ok=True)
    return d

def _save_file(bytes_data: bytes, original_name: str) -> Tuple[str, str]:
    h = _sha256(bytes_data)
    d = _file_dir(h)
    meta_path = os.path.join(d, "meta.json")
    file_path = os.path.join(d, "file")
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(bytes_data)
        meta = {
            "hash": h,
            "original_name": original_name,
            "safe_name": _safe_name(original_name),
            "size": len(bytes_data),
            "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    else:
        if not os.path.exists(meta_path):
            meta = {
                "hash": h,
                "original_name": original_name,
                "safe_name": _safe_name(original_name),
                "size": len(bytes_data),
                "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
    return h, file_path

def _list_saved_files() -> List[dict]:
    items = []
    for a in os.listdir(UPLOAD_DIR):
        pa = os.path.join(UPLOAD_DIR, a)
        if not os.path.isdir(pa) or a == "bundles":
            continue
        for b in os.listdir(pa):
            pb = os.path.join(pa, b)
            if not os.path.isdir(pb):
                continue
            for c in os.listdir(pb):
                pc = os.path.join(pb, c)
                if not os.path.isdir(pc):
                    continue
                meta_path = os.path.join(pc, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            items.append(json.load(f))
                    except Exception:
                        pass
    items.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    return items

def _compute_bundle_hash(file_hashes: List[str]) -> str:
    joined = "|".join(sorted(file_hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()

def _bundle_cached(bundle_hash: str) -> dict | None:
    d = _bundle_dir(bundle_hash)
    rp = os.path.join(d, "result.json")
    if os.path.exists(rp):
        with open(rp, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# JSON-sanitize before writing to disk
def _save_bundle_result(bundle_hash: str, result: dict, file_hashes: List[str], file_names: List[str]) -> None:
    d = _bundle_dir(bundle_hash)
    payload = {
        "bundle_hash": bundle_hash,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "file_hashes": file_hashes,
        "file_names": file_names,
        "result": None,  # filled below with json-safe copy
    }
    payload["result"] = clean_for_json(result)
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _list_bundles() -> List[dict]:
    items = []
    for name in os.listdir(BUNDLES_DIR):
        p = os.path.join(BUNDLES_DIR, name, "result.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    items.append({
                        "bundle_hash": obj.get("bundle_hash", name),
                        "created_at": obj.get("created_at"),
                        "file_names": obj.get("file_names", []),
                        "file_hashes": obj.get("file_hashes", []),
                    })
            except Exception:
                pass
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items

# ----------------------- Utils -----------------------
def clean_for_json(data):
    # Dates / Timestamps -> ISO date string
    if isinstance(data, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64)):
        try:
            return pd.to_datetime(data).date().isoformat()
        except Exception:
            return str(data)

    # Numpy numerics
    if isinstance(data, (np.int64, np.int32)):
        return int(data)
    if isinstance(data, (np.float64, np.float32, float)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)

    # Containers
    if isinstance(data, list):
        return [clean_for_json(x) for x in data]
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}

    # Pandas NA
    try:
        if pd.isna(data):
            return None
    except Exception:
        pass

    return data

_SPONSOR_WORDS = {
    "ladbrokes","sportsbet","bet365","tabtouch","tab","william hill",
    "unibet","betfair","palmerbet","neds","bluebet","pointbet","pointsbet","sportsbet-ballarat"
}

def _strip_accents_punct(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _remove_country_suffix(s):
    return re.sub(r"\s*\([^)]{2,3}\)\s*$", "", s)

def token_fingerprint(s):
    if not isinstance(s, str): s = str(s)
    s = s.lower().strip()
    s = _remove_country_suffix(s)
    s = re.sub(r"^\d+\.\s*", "", s)
    s = _strip_accents_punct(s)
    tokens = [t for t in s.split() if t not in _SPONSOR_WORDS]
    return " ".join(sorted(tokens))

def standardize_track(name):
    if not isinstance(name, str): return name
    s = name.lower().strip()
    is_synth = "synthetic" in s
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    for w in _SPONSOR_WORDS:
        s = s.replace(w, " ")
    s = _strip_accents_punct(s).lower()
    replacements = {"ladbrokes cannon park": "cannon park", "cannon park": "cairns"}
    s = replacements.get(s, s)
    s = re.sub(r"\s+", " ", s).strip()
    if is_synth and "ballarat" in s and "synthetic" not in s:
        s = f"{s} synthetic"
    return s

def standardize_horse(name):
    if not isinstance(name, str): return name
    s = name.strip()
    s = _remove_country_suffix(s)
    s = re.sub(r"^\d+\.\s*", "", s)
    s = _strip_accents_punct(s).lower()
    return re.sub(r"\s+", " ", s).strip()

def extract_race_num(x):
    if pd.isna(x): return np.nan
    s = str(x)
    m = re.search(r"(?i)\bR(?:ace)?\s*0*([1-9]\d*)\b", s)
    if not m: m = re.search(r"\b([1-9]\d*)\b", s)
    return float(m.group(1)) if m else np.nan

def robust_to_datetime(series, dayfirst=False):
    def _try(ts, **kw): return pd.to_datetime(ts, errors="coerce", **kw)
    out = _try(series, dayfirst=dayfirst, utc=False)
    if hasattr(out, "dt"): out = out.dt.date
    mask = pd.isna(out)
    if hasattr(mask, "mean") and mask.mean() > 0.3:
        alt = _try(series[mask], dayfirst=not dayfirst, utc=False)
        if hasattr(alt, "dt"): alt = alt.dt.date
        try: out.loc[mask] = alt
        except Exception: pass
    mask = pd.isna(out)
    if hasattr(mask, "any") and mask.any():
        num = pd.to_numeric(series[mask], errors="coerce")
        sec = _try(num, unit="s", utc=False);  sec = sec.dt.date if hasattr(sec, "dt") else sec
        try: out.loc[mask & pd.notna(sec)] = sec
        except Exception: pass
        ms = _try(num, unit="ms", utc=False); ms = ms.dt.date if hasattr(ms, "dt") else ms
        try: out.loc[mask & pd.notna(ms)] = ms
        except Exception: pass
    return out

def as_numeric_safe(obj):
    if isinstance(obj, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        x = pd.to_numeric(obj, errors="coerce")
        if not isinstance(x, (pd.Series, pd.Index)):
            x = pd.Series(x)
        return x.replace([np.inf, -np.inf], np.nan)
    try:
        x = float(obj)
    except (TypeError, ValueError):
        return np.nan
    if np.isinf(x) or np.isnan(x):
        return np.nan
    return x

def map_fuzzy_to_known(values, known, cutoff=0.90):
    known_set = set(known)
    out = []
    for v in values.fillna(""):
        if v in known_set:
            out.append(v)
        else:
            match = get_close_matches(v, known, n=1, cutoff=cutoff)
            out.append(match[0] if match else v)
    return pd.Series(out, index=values.index)

# ------------------- Betfair detection & transform -------------------
def looks_like_betfair_prices(df: pd.DataFrame, fname: str) -> bool:
    low = {c.lower() for c in df.columns}
    hints = {"selection_name", "menu_hint", "event_name", "marketstarttime", "bsp", "sp", "event_dt"}
    return bool(hints & low) or "prices" in fname or "betfair" in fname

def transform_betfair_prices_to_internal(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    def col(name): return lower_map.get(name.lower())

    out = pd.DataFrame()
    # Horse
    for cand in ["selection_name","runner_name","horse_name","selection"]:
        c = col(cand)
        if c:
            out["Horse Name"] = df[c].astype(str)
            break
    if "Horse Name" not in out.columns: out["Horse Name"] = ""
    # Track
    if col("menu_hint"):
        out["Track"] = df[col("menu_hint")].astype(str).str.split(r" \(", n=1, regex=True).str[0]
    elif col("event_name"):
        trk = df[col("event_name")].astype(str)
        trk = trk.str.replace(r"\s*\(.*\)", "", regex=True)
        trk = trk.str.replace(r"\sR(?:ace)?\s*\d+.*$", "", regex=True)
        out["Track"] = trk
    else:
        out["Track"] = ""
    # Race number
    out["race_num"] = np.nan
    for cand in ["event_name","Race","race","market_name"]:
        c = col(cand)
        if c:
            out["race_num"] = df[c].apply(extract_race_num)
            break
    # Date
    set_date = False
    for cand in ["event_dt","event_date","event_time","marketStartTime","Date","timestamp","marketstarttime"]:
        c = col(cand)
        if c:
            out["Date"] = pd.to_datetime(df[c], errors="coerce", dayfirst=True).dt.date
            set_date = True
            break
    if not set_date:
        m = re.search(r"(\d{8})", fname)
        out["Date"] = pd.to_datetime(m.group(1), format="%d%m%Y", errors="coerce").date() if m else pd.NaT
    # Prices (BSP)
    if col("bsp"):
        out["bsp"] = as_numeric_safe(df[col("bsp")])
    elif col("sp"):
        out["bsp"] = as_numeric_safe(df[col("sp")])
    else:
        out["bsp"] = np.nan
    if pd.isna(out["bsp"]).all():
        for cand in ["last_price_traded","ltp","price"]:
            c = col(cand)
            if c:
                out["bsp"] = as_numeric_safe(df[c]); break
    # Morning WAP
    out["morningwap"] = as_numeric_safe(df[col("morningwap")]) if col("morningwap") else as_numeric_safe(pd.Series(dtype=float))
    # Result flag
    wl = df.get(col("win_lose")) if col("win_lose") else None
    if wl is not None:
        out["win_lose"] = pd.to_numeric(wl, errors="coerce").fillna(0).astype(int)
    else:
        out["win_lose"] = np.nan
    # Field size derivation
    field_cols = ["numberofrunners","runners","runnercount","maxrunners","field_size"]
    fs = None
    for cand in field_cols:
        chosen = col(cand)
        if chosen is not None:
            fs = as_numeric_safe(df[chosen])
            break
    if fs is None:
        ev = col("event_id") or col("market_id") or col("marketid")
        sel = col("selection_id") or col("runner_id") or col("selectionid")
        if ev and sel:
            counts = df.groupby(df[ev])[sel].nunique()
            fs = df[ev].map(counts).astype(float)
    out["field_size"] = fs if fs is not None else as_numeric_safe(pd.Series(dtype=float))
    # Standardise & fingerprints
    out["Track"] = out["Track"].astype(str).map(standardize_track)
    out["Horse Name"] = out["Horse Name"].astype(str).map(standardize_horse)
    out["track_fp"] = out["Track"].map(token_fingerprint)
    out["horse_fp"] = out["Horse Name"].map(token_fingerprint)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.date
    cols = ["Date","Track","Horse Name","race_num","bsp","morningwap","win_lose","track_fp","horse_fp","field_size"]
    for c in cols:
        if c not in out.columns: out[c] = np.nan
    return out[cols]

# ---------------------- Matching ----------------------
def match_win_prices_to_tips(tips_df: pd.DataFrame, win_df: pd.DataFrame) -> pd.DataFrame:
    if win_df is None or win_df.empty:
        tmp = tips_df.copy()
        for c in ["bsp","morningwap","win_lose","Date_win","field_size"]:
            tmp[c] = np.nan
        return tmp
    tips = tips_df.rename(columns={"Date": "Date_tip"}).copy()
    win_df = win_df.rename(columns={"Date": "Date_win"}).copy()
    
    # Ensure join keys exist and are of the same type before the fuzzy merge
    for col in ["track_fp", "horse_fp"]:
        if col not in tips.columns:
            tips[col] = ""
        if col not in win_df.columns:
            win_df[col] = ""
        tips[col] = tips[col].astype(str)
        win_df[col] = win_df[col].astype(str)
    
    # Handle the fuzzy merge case
    if ("race_num" in tips.columns and "race_num" in win_df.columns and
        tips["race_num"].isna().mean() > 0.9 and win_df["race_num"].isna().mean() > 0.9):
        fb = tips.merge(
            win_df[["track_fp","horse_fp","bsp","morningwap","win_lose","Date_win","field_size"]],
            on=["track_fp","horse_fp"], how="left"
        )
        fb["Date_tip"] = pd.to_datetime(fb["Date_tip"], errors="coerce")
        fb["Date_win"] = pd.to_datetime(fb["Date_win"], errors="coerce")
        fb["date_diff"] = (fb["Date_win"] - fb["Date_tip"]).abs().dt.days
        fb = fb.loc[(fb["date_diff"].isna()) | (fb["date_diff"] <= 2)]
        fb = fb.sort_values(["Date_tip","date_diff"]).drop_duplicates(
            subset=["Date_tip","track_fp","horse_fp"], keep="first"
        )
        return fb
    
    # Ensure join keys exist and are of the same type for the standard merge
    join_cols = [c for c in ["track_fp","race_num","horse_fp"] if c in tips.columns and c in win_df.columns]
    
    if not join_cols:
        tmp = tips.copy()
        for c in ["bsp","morningwap","win_lose","Date_win","field_size"]:
            tmp[c] = np.nan
        return tmp

    for col in join_cols:
        tips[col] = tips[col].astype(str)
        win_df[col] = win_df[col].astype(str)

    merged = tips.merge(win_df, on=join_cols, how="left")
    return merged

# -------------------- Main analysis --------------------
def perform_full_analysis(dataframes: Dict) -> Dict:
    response = {"daily_summary": [], "charts": {}}
    tips_df = dataframes.get("tips")
    
    # Initialize the variable to None to prevent the NameError
    race_data_df = dataframes.get("race_data")
    
    win_prices_df = dataframes.get("win_prices")

    chart_keys = [
        "cumulative_profit","rolling_roi","roi_by_tipster","roi_by_odds",
        "price_movement_histogram","clv_trend","win_rate_vs_field_size"
    ]
    for k in chart_keys:
        response["charts"][k] = (
            {"labels": [], "datasets": []}
            if k in {"cumulative_profit","rolling_roi","clv_trend"} else {"labels": [], "data": []}
        )
    if tips_df is None or tips_df.empty:
        return response

    # Merge base with race data (if any)
    if race_data_df is not None and not race_data_df.empty:
        # Before merging, ensure columns exist in race_data_df
        if "track_fp" not in race_data_df.columns:
            race_data_df["track_fp"] = np.nan
        if "horse_fp" not in race_data_df.columns:
            race_data_df["horse_fp"] = np.nan
        if "race_num" not in race_data_df.columns:
            race_data_df["race_num"] = np.nan
        if "field_size" not in race_data_df.columns:
            race_data_df["field_size"] = np.nan
            
        # The fix: Explicitly convert the merge keys to string type
        # to ensure consistency and prevent the ValueError.
        tips_df["track_fp"] = tips_df["track_fp"].astype(str)
        race_data_df["track_fp"] = race_data_df["track_fp"].astype(str)
        tips_df["horse_fp"] = tips_df["horse_fp"].astype(str)
        race_data_df["horse_fp"] = race_data_df["horse_fp"].astype(str)
        tips_df["race_num"] = tips_df["race_num"].astype(str)
        race_data_df["race_num"] = race_data_df["race_num"].astype(str)

        merged = tips_df.merge(
            race_data_df[["track_fp","race_num","horse_fp","BestOdds","field_size"]],
            on=["track_fp","race_num","horse_fp"], how="left"
        )
    else:
        merged = tips_df.copy()
        merged["BestOdds"] = np.nan
        if "field_size" not in merged.columns:
            merged["field_size"] = np.nan

    # If tips came from prices, carry price columns along
    if dataframes.get("_tips_from_prices"):
        merged.rename(columns={"Date":"Date_tip"}, inplace=True)
        merged["Date_win"] = merged["Date_tip"]
    else:
        if win_prices_df is not None and not win_prices_df.empty:
            merged = match_win_prices_to_tips(merged, win_prices_df)
        else:
            merged["bsp"] = merged["morningwap"] = merged["win_lose"] = merged["Date_win"] = np.nan

    orig_date = pd.to_datetime(merged.get("Date"), errors="coerce").dt.date if "Date" in merged.columns else pd.NaT
    dw = pd.to_datetime(merged.get("Date_win"), errors="coerce").dt.date
    merged["Date"] = np.where(pd.notna(dw), dw, orig_date)

    for c in ["bsp","morningwap","BestOdds","win_lose","field_size"]:
        if c in merged.columns:
            merged[c] = as_numeric_safe(merged[c])
        else:
            merged[c] = np.nan
    merged["win_lose"] = merged["win_lose"].fillna(0)

    merged["BestOdds_eff"] = np.where(merged["BestOdds"] > 1, merged["BestOdds"], merged["morningwap"])

    merged["Profit"] = np.where(merged["win_lose"] == 1, merged["bsp"], 0) - 1
    merged.loc[merged["bsp"].isna(), "Profit"] = -1

    for d, g in merged.groupby("Date", dropna=False):
        bets = len(g)
        rtn = g.loc[g["win_lose"] == 1, "bsp"].fillna(0).sum()
        denom = max(bets, 1)
        valid_clv = (g["bsp"] > 1) & (g["BestOdds_eff"] > 1)
        valid_mw = (g["morningwap"] > 1) & (g["bsp"] > 0)
        response["daily_summary"].append({
            "Date": str(d),
            "Bets Placed": bets,
            "Units Staked": bets,
            "Units Returned": float(rtn),
            "ROI %": (float(rtn - bets) / denom) * 100,
            "Win Rate %": g["win_lose"].fillna(0).mean() * 100,
            "Avg Odds": float(g["bsp"].dropna().mean()) if not pd.isna(g["bsp"].dropna().mean()) else 0,
            "CLV": float(((g.loc[valid_clv, "bsp"] / g.loc[valid_clv, "BestOdds_eff"] - 1).mean() * 100) if valid_clv.any() else 0),
            "Drifters %": float(((g.loc[valid_mw, "bsp"] > g.loc[valid_mw, "morningwap"]).sum() / max(valid_mw.sum(), 1)) * 100),
            "Steamers %": float(((g.loc[valid_mw, "bsp"] < g.loc[valid_mw, "morningwap"]).sum() / max(valid_mw.sum(), 1)) * 100),
        })

    merged["Date"] = pd.to_datetime(merged["Date"]).dt.date
    if "Tip Website" not in merged.columns:
        merged["Tip Website"] = "DefaultTipster"

    # Charts
    dp = merged.groupby(["Tip Website","Date"], as_index=False)["Profit"].sum().sort_values(["Tip Website","Date"])
    pivot = dp.pivot(index="Date", columns="Tip Website", values="Profit").fillna(0).cumsum()
    response["charts"]["cumulative_profit"] = {
        "labels": [d.strftime("%Y-%m-%d") for d in pivot.index],
        "datasets": [{"name": c, "data": pivot[c].round(4).tolist()} for c in pivot.columns]
    }
    if not response["charts"]["cumulative_profit"]["datasets"]:
        numerics = merged.select_dtypes(include='number').columns
        response["charts"]["cumulative_profit"]["datasets"] = [
            {"name": col, "data": merged[col].fillna(0).cumsum().tolist()} for col in numerics
        ]
        response["charts"]["cumulative_profit"]["labels"] = [str(i) for i in range(len(merged))]

    ds = merged.groupby(["Tip Website","Date"]).agg(
        bets=("Profit","size"),
        profit=("Profit","sum"),
    ).reset_index()
    out = []
    for tip, g in ds.groupby("Tip Website"):
        g = g.sort_values("Date").set_index("Date")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx, fill_value=0)
        g["bets_roll"] = g["bets"].rolling(30, min_periods=1).sum()
        g["profit_roll"] = g["profit"].rolling(30, min_periods=1).sum()
        g["roi30"] = np.where(g["bets_roll"] > 0, (g["profit_roll"] / g["bets_roll"]) * 100.0, 0.0)
        g["Tip Website"] = tip
        g["Date"] = g.index.date
        out.append(g.reset_index(drop=True)[["Tip Website","Date","roi30"]])
    if out:
        rr = pd.concat(out)
        p = rr.pivot(index="Date", columns="Tip Website", values="roi30").ffill().fillna(0)
        response["charts"]["rolling_roi"] = {
            "labels": [d.strftime("%Y-%m-%d") for d in p.index],
            "datasets": [{"name": c, "data": p[c].round(4).tolist()} for c in p.columns]
        }
    if not response["charts"]["rolling_roi"].get("datasets"):
        response["charts"]["rolling_roi"] = {
            "labels": response["charts"]["cumulative_profit"]["labels"],
            "datasets": [{"name": "ROI", "data": [0] * len(response["charts"]["cumulative_profit"]["labels"])}]
        }

    rbt = merged.groupby("Tip Website")["Profit"].mean().fillna(0) * 100
    response["charts"]["roi_by_tipster"] = {"labels": rbt.index.tolist(), "data": rbt.round(4).tolist()}

    merged["bsp"] = as_numeric_safe(merged.get("bsp"))
    merged["odds_bin"] = pd.cut(
        merged["bsp"].where(merged["bsp"] > 1),
        bins=[1, 3, 5, 10, 20, 50, 1000],
        labels=["$1-3","$3-5","$5-10","$10-20","$20-50","$50+"],
        include_lowest=True
    ).astype(str)
    rob = merged.groupby("odds_bin")["Profit"].mean().fillna(0) * 100
    response["charts"]["roi_by_odds"] = {"labels": rob.index.tolist(), "data": rob.round(4).tolist()}

    pm = (merged["bsp"] > 1) & (merged["morningwap"] > 1)
    pmv = ((merged.loc[pm, "bsp"] - merged.loc[pm, "morningwap"]) / merged.loc[pm, "morningwap"]).astype(float)
    if pmv.empty:
        pmv = pd.Series([0.0])
    pmv = pmv.clip(-0.8, 1.5)
    response["charts"]["price_movement_histogram"] = {"data": pmv.round(6).tolist()}

    valid = (merged["bsp"] > 1) & (merged["BestOdds_eff"] > 1)
    merged["clv_calc"] = ((merged["bsp"] / merged["BestOdds_eff"]) - 1) * 100
    clvt = merged.loc[valid].groupby("Date")["clv_calc"].mean()
    if clvt.empty:
        response["charts"]["clv_trend"] = {
            "labels": response["charts"]["cumulative_profit"]["labels"],
            "datasets": [{"name": "CLV", "data": [0] * len(response["charts"]["cumulative_profit"]["labels"])}]
        }
    else:
        response["charts"]["clv_trend"] = {
            "labels": [d.strftime("%Y-%m-%d") for d in clvt.index],
            "datasets": [{"name": "CLV", "data": clvt.round(4).tolist()}]
        }

    if "field_size" in merged.columns and merged["field_size"].notna().any():
        wr = merged.dropna(subset=["field_size"]).copy()
        wr["field_size"] = wr["field_size"].round(0).astype(int)
        grp = wr.groupby("field_size")["win_lose"].mean().fillna(0) * 100
        response["charts"]["win_rate_vs_field_size"] = {
            "labels": [str(i) for i in grp.index],
            "data": grp.round(4).tolist()
        }
    else:
        response["charts"]["win_rate_vs_field_size"] = {"labels": ["0"], "data": [0]}

    # Final step: Ensure all values are JSON-safe before returning.
    return clean_for_json(response)

# ----------------------- API -----------------------
@app.post("/analyze/")
async def analyze_betting_files(files: List[UploadFile] = File(...)):
    try:
        # 1) Read bytes, compute per-file hashes, persist originals
        bytes_list = []
        original_names = []
        file_hashes = []
        for f in files:
            data = await f.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"Empty file: {f.filename or 'uploaded'}")
            bytes_list.append(data)
            original_names.append(f.filename or "uploaded")
            h, _ = _save_file(data, f.filename or "uploaded")
            file_hashes.append(h)

        # 2) Compute bundle hash and return cached result if present
        bundle_hash = _compute_bundle_hash(file_hashes)
        cached = _bundle_cached(bundle_hash)
        if cached is not None:
            return JSONResponse(cached)

        # 3) Not cached: parse & analyze
        dataframes: Dict[str, pd.DataFrame] = {}
        for data, fname in zip(bytes_list, original_names):
            lowname = (fname or "uploaded").lower()
            if lowname.endswith(".csv"):
                raw = pd.read_csv(io.BytesIO(data))
            elif lowname.endswith(".xlsx"):
                raw = pd.read_excel(io.BytesIO(data))
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {fname}")

            if looks_like_betfair_prices(raw, lowname):
                win_df = transform_betfair_prices_to_internal(raw, lowname)
                dataframes["win_prices"] = pd.concat(
                    [dataframes.get("win_prices", pd.DataFrame()), win_df],
                    ignore_index=True
                )
                tips_new = win_df[[
                    "Date","Track","Horse Name","race_num",
                    "track_fp","horse_fp","bsp","morningwap","win_lose","field_size"
                ]].copy()
                tips_new["Tip Website"] = "Betfair"
                if "tips" not in dataframes:
                    tips_new["tip_id"] = np.arange(len(tips_new))
                    dataframes["tips"] = tips_new
                    dataframes["_tips_from_prices"] = True
                elif dataframes.get("_tips_from_prices", False):
                    start = int(dataframes["tips"]["tip_id"].max()) + 1 if "tip_id" in dataframes["tips"].columns else 0
                    tips_new["tip_id"] = np.arange(start, start + len(tips_new))
                    dataframes["tips"] = pd.concat([dataframes["tips"], tips_new], ignore_index=True)

            elif "tips" in lowname:
                df = raw.copy()
                if "Track" in df.columns:
                    df["Track"] = df["Track"].astype(str).map(standardize_track)
                    df["track_fp"] = df["Track"].map(token_fingerprint)
                if "Horse Name" in df.columns:
                    df["Horse Name"] = df["Horse Name"].astype(str).map(standardize_horse)
                    df["horse_fp"] = df["Horse Name"].map(token_fingerprint)
                if "Date" in df.columns:
                    df["Date"] = robust_to_datetime(df["Date"], dayfirst=True)
                dataframes["tips"] = df
                dataframes["_tips_from_prices"] = False

            elif "race" in lowname:
                dataframes["race_data"] = raw.copy()
                if "Track" in dataframes["race_data"].columns:
                    dataframes["race_data"]["Track"] = dataframes["race_data"]["Track"].astype(str).map(standardize_track)
                    dataframes["race_data"]["track_fp"] = dataframes["race_data"]["Track"].map(token_fingerprint)
                if "Horse Name" in dataframes["race_data"].columns:
                    dataframes["race_data"]["Horse Name"] = dataframes["race_data"]["Horse Name"].astype(str).map(standardize_horse)
                    dataframes["race_data"]["horse_fp"] = dataframes["race_data"]["Horse Name"].map(token_fingerprint)
                if "race_num" in dataframes["race_data"].columns:
                    dataframes["race_data"]["race_num"] = dataframes["race_data"]["race_num"].apply(extract_race_num)
                if "field_size" not in dataframes["race_data"].columns:
                    dataframes["race_data"]["field_size"] = np.nan
                
            elif "win" in lowname or "prices" in lowname:
                dataframes["win_prices"] = transform_betfair_prices_to_internal(raw, lowname)

            else:
                if "tips" not in dataframes:
                    df = raw.copy()
                    df["Tip Website"] = df.get("Tip Website", "DefaultTipster")
                    df["tip_id"] = np.arange(len(df))
                    if "Track" in df.columns:
                        df["Track"] = df["Track"].astype(str).map(standardize_track)
                        df["track_fp"] = df["Track"].map(token_fingerprint)
                    if "Horse Name" in df.columns:
                        df["Horse Name"] = df["Horse Name"].astype(str).map(standardize_horse)
                        df["horse_fp"] = df["Horse Name"].map(token_fingerprint)
                    if "Date" in df.columns:
                        df["Date"] = robust_to_datetime(df["Date"], dayfirst=True)
                    dataframes["tips"] = df
                    dataframes["_tips_from_prices"] = False

        if "tips" not in dataframes or dataframes["tips"].empty:
            raise HTTPException(status_code=400, detail="No tips data found or derivable from upload.")
        
        # Ensure tips_df has all the necessary fingerprint columns before analysis
        if "track_fp" not in dataframes["tips"].columns:
            dataframes["tips"]["track_fp"] = dataframes["tips"].get("Track", pd.Series([np.nan]*len(dataframes["tips"]))).astype(str).map(token_fingerprint)
        if "horse_fp" not in dataframes["tips"].columns:
            dataframes["tips"]["horse_fp"] = dataframes["tips"].get("Horse Name", pd.Series([np.nan]*len(dataframes["tips"]))).astype(str).map(token_fingerprint)
        if "race_num" not in dataframes["tips"].columns:
            dataframes["tips"]["race_num"] = dataframes["tips"].get("race_num", pd.Series([np.nan]*len(dataframes["tips"])))
        if "field_size" not in dataframes["tips"].columns:
            dataframes["tips"]["field_size"] = np.nan

        result = perform_full_analysis(dataframes)

        # ---- Build a JSON-safe copy of tips for the frontend
        tips_copy = dataframes["tips"].copy()
        for col in tips_copy.columns:
            if (pd.api.types.is_datetime64_any_dtype(tips_copy[col]) or
                pd.api.types.is_datetime64tz_dtype(tips_copy[col])):
                tips_copy[col] = pd.to_datetime(tips_copy[col], errors="coerce").dt.date.astype(str)
            else:
                if tips_copy[col].dtype == "object":
                    tips_copy[col] = tips_copy[col].apply(
                        lambda x: pd.to_datetime(x).date().isoformat()
                        if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date, np.datetime64))
                        else x
                    )

        safe_result = clean_for_json(result)
        safe_result["data"] = (
            tips_copy.replace([np.inf, -np.inf], np.nan)
                     .where(pd.notna(tips_copy), None)
                     .to_dict(orient="records")
        )

        # 4) Save result for future reuse
        _save_bundle_result(bundle_hash, safe_result, file_hashes, original_names)

        return {
            "bundle_hash": bundle_hash,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "file_hashes": file_hashes,
            "file_names": original_names,
            "result": safe_result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ----------------------- Saved items endpoints -----------------------
@app.get("/saved/files")
async def list_saved_files():
    return _list_saved_files()

@app.get("/saved/files/{file_hash}/download")
async def download_saved_file(file_hash: str):
    d = _file_dir(file_hash)
    p = os.path.join(d, "file")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="File not found")
    meta_path = os.path.join(d, "meta.json")
    filename = f"{file_hash}.bin"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                filename = meta.get("safe_name", filename)
        except Exception:
            pass
    return FileResponse(p, filename=filename)

@app.get("/saved/bundles")
async def list_saved_bundles():
    return _list_bundles()

@app.get("/saved/bundles/{bundle_hash}")
async def get_bundle_result(bundle_hash: str):
    cached = _bundle_cached(bundle_hash)
    if cached is None:
        raise HTTPException(status_code=404, detail="Bundle not found")
    return cached

# ----------------------- Health -----------------------
@app.get("/health")
async def health():
    return {"status": "healthy"}

# ----------------------- Static frontend (optional) -----------------------
script_dir = os.path.dirname(__file__)
frontend_dir = os.path.join(os.path.dirname(script_dir), "frontend")
if os.path.exists(frontend_dir):
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse(os.path.join(frontend_dir, 'favicon_io', 'favicon.ico'))
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    @app.get("/", response_class=FileResponse)
    async def root():
        return FileResponse(os.path.join(frontend_dir, 'index.html'))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)