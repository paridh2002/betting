from typing import Dict, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
import io, math, os, logging, traceback, re, unicodedata
from datetime import datetime
from difflib import get_close_matches
import uvicorn
import webbrowser
import threading
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, MetaData, Table

# ---------- FastAPI setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Advanced Betting Performance Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "sqlite:///betting_analysis.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# Define the table structure for daily_stats
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
# Create the database schema
metadata.create_all(engine)

# ---------- Utilities ----------
def clean_for_json(data):
    """Make numpy/pandas types serializable."""
    if isinstance(data, (np.int64, np.int32)): return int(data)
    if isinstance(data, (np.float64, np.float32)):
        return float(data) if not (math.isnan(data) or math.isinf(data)) else 0
    if isinstance(data, list): return [clean_for_json(x) for x in data]
    if isinstance(data, dict): return {k: clean_for_json(v) for k, v in data.items()}
    if pd.isna(data): return 0
    return data

_SPONSOR_WORDS = {"ladbrokes","sportsbet","bet365","tabtouch","tab","william hill",
                  "unibet","betfair","palmerbet","neds","bluebet","pointbet","pointsbet","sportsbet-ballarat"}

def _strip_accents_punct(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _remove_country_suffix(s):
    return re.sub(r"\s*\((?:[A-Z]{2,3})\)\s*$", "", s)

def token_fingerprint(s):
    s = s.lower().strip()
    s = _remove_country_suffix(s)
    s = re.sub(r"^\d+\.?\s*", "", s)
    s = _strip_accents_punct(s)
    tokens = [t for t in s.split() if t not in _SPONSOR_WORDS]
    return " ".join(sorted(tokens))

def standardize_track(name):
    if not isinstance(name, str): return name
    s = name.lower().strip()
    is_synth = "synthetic" in s
    s = re.sub(r"\s*\(.?\)\s", " ", s)
    for w in _SPONSOR_WORDS: s = s.replace(w, " ")
    s = _strip_accents_punct(s).lower()
    replacements = {"ladbrokes cannon park":"cannon park", "cannon park":"cairns"}
    s = replacements.get(s, s)
    s = re.sub(r"\s+", " ", s).strip()
    if is_synth and "ballarat" in s and "synthetic" not in s:
        s = f"{s} synthetic"
    return s

def standardize_horse(name):
    if not isinstance(name, str): return name
    s = name.strip()
    s = _remove_country_suffix(s)
    s = re.sub(r"^\d+\.?\s*", "", s)
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
    out = _try(series, dayfirst=dayfirst, utc=False).dt.date
    mask = pd.isna(out)
    if mask.mean() > 0.3:
        alt = _try(series[mask], dayfirst=not dayfirst, utc=False).dt.date
        out.loc[mask] = alt
    mask = pd.isna(out)
    if mask.any():
        num = pd.to_numeric(series[mask], errors="coerce")
        sec = _try(num, unit="s", utc=False).dt.date
        out.loc[mask & sec.notna()] = sec
        ms = _try(num, unit="ms", utc=False).dt.date
        out.loc[mask & ms.notna()] = ms
    return out

def as_numeric_safe(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)

def map_fuzzy_to_known(values, known, cutoff=0.90):
    known_set = set(known)
    out = []
    for v in values.fillna(""):
        if v in known_set: out.append(v)
        else:
            match = get_close_matches(v, known, n=1, cutoff=cutoff)
            out.append(match[0] if match else v)
    return pd.Series(out, index=values.index)

# ---------- Cleaning / matching ----------
def clean_and_standardize_data(df: pd.DataFrame, file_type: str, known_tracks=None, known_horses=None) -> pd.DataFrame:
    df.rename(columns=lambda c: c.strip(), inplace=True)
    if file_type == "tips":
        if "Track" not in df.columns and "track" in df.columns:
            df.rename(columns={"track":"Track"}, inplace=True)
        if "First Selection Name" in df.columns and "Horse Name" not in df.columns:
            df["Horse Name"] = df["First Selection Name"]
        date_col = "Scrape Date" if "Scrape Date" in df.columns else ("Date" if "Date" in df.columns else None)
        if not date_col: raise HTTPException(status_code=400, detail="Tips file missing date column.")
        df["Date"] = robust_to_datetime(df[date_col])
        df["Track"] = df["Track"].astype(str).map(standardize_track)
        df["Horse Name"] = df["Horse Name"].astype(str).map(standardize_horse)
        df["race_num"] = df["Race"].apply(extract_race_num) if "Race" in df.columns else np.nan
        df["track_fp"] = df["Track"].map(token_fingerprint)
        df["horse_fp"] = df["Horse Name"].map(token_fingerprint)
        df["Tip Website"] = df.get("Tip Website", "Unknown")
        df["tip_id"] = np.arange(len(df))
        return df[["tip_id", "Date", "Tip Website", "Track", "Horse Name", "race_num", "track_fp", "horse_fp"]]

    elif file_type == "race_data":
        ren = {"HorseName":"Horse Name","RaceTrack":"Track","RaceNum":"Race"}
        df.rename(columns={k:v for k,v in ren.items() if k in df.columns}, inplace=True)
        df["Track"] = df["Track"].astype(str).map(standardize_track)
        df["Horse Name"] = df["Horse Name"].astype(str).map(standardize_horse)
        df["race_num"] = df["Race"].apply(extract_race_num) if "Race" in df.columns else np.nan
        df["BestOdds"] = as_numeric_safe(df.get("BestOdds", np.nan))
        df["track_fp"] = df["Track"].map(token_fingerprint)
        df["horse_fp"] = df["Horse Name"].map(token_fingerprint)
        df["field_size"] = df.groupby(["track_fp","race_num"])["Horse Name"].transform("count")
        if known_tracks and known_horses:
            df["track_fp"] = map_fuzzy_to_known(df["track_fp"], known_tracks, cutoff=0.90)
            df["horse_fp"] = map_fuzzy_to_known(df["horse_fp"], known_horses, cutoff=0.90)
        return df[["Track","Horse Name","race_num","BestOdds","field_size","track_fp","horse_fp"]]

    elif file_type == "prices":
        if "selection_name" in df.columns and "Horse Name" not in df.columns:
            df.rename(columns={"selection_name":"Horse Name"}, inplace=True)
        if "menu_hint" in df.columns:
            df["Track"] = df["menu_hint"].astype(str).str.split(r" \(").str[0].map(standardize_track)
        elif "RaceTrack" in df.columns:
            df["Track"] = df["RaceTrack"].astype(str).map(standardize_track)
        else:
            df["Track"] = df.get("Track","").astype(str).map(standardize_track)
        if "event_name" in df.columns:
            df["race_num"] = df["event_name"].apply(extract_race_num)
        elif "Race" in df.columns:
            df["race_num"] = df["Race"].apply(extract_race_num)
        else:
            df["race_num"] = np.nan
        df["Horse Name"] = df["Horse Name"].astype(str).str.replace(r"^\d+\.?\s*","", regex=True).map(standardize_horse)
        date_source=None
        for cand in ["event_dt","event_date","event_time","marketStartTime","Date"]:
            if cand in df.columns: date_source=cand; break
        if not date_source: raise HTTPException(status_code=400, detail="Prices file missing an event date column.")
        df["Date"] = robust_to_datetime(df[date_source], dayfirst=True)
        for col in ["bsp","morningwap","win_lose"]:
            if col in df.columns: df[col]=as_numeric_safe(df[col])
        if "win_lose" in df.columns and df["win_lose"].isna().mean()>0.5:
            wl=df["win_lose"].astype(str).str.lower()
            df["win_lose"]=np.where(wl.str.contains("win|1|true"),1,
                                     np.where(wl.str.contains("lose|0|false"),0,np.nan))
        df["track_fp"]=df["Track"].map(token_fingerprint)
        df["horse_fp"]=df["Horse Name"].map(token_fingerprint)
        for c in ["bsp","morningwap"]:
            df.loc[df[c]<=1,c]=np.nan
        if known_tracks and known_horses:
            df["track_fp"] = map_fuzzy_to_known(df["track_fp"], known_tracks, cutoff=0.90)
            df["horse_fp"] = map_fuzzy_to_known(df["horse_fp"], known_horses, cutoff=0.90)
        return df[["Date", "Track", "Horse Name", "race_num", "bsp", "morningwap", "win_lose", "track_fp", "horse_fp"]]
    
    return pd.DataFrame() # Return empty if file_type is unknown

def match_win_prices_to_tips(tips_df: pd.DataFrame, win_df: pd.DataFrame) -> pd.DataFrame:
    if win_df is None or win_df.empty:
        tmp = tips_df.copy()
        for c in ["bsp","morningwap","win_lose","Date_win"]: tmp[c]=np.nan
        return tmp

    tips = tips_df.rename(columns={"Date":"Date_tip"})
    win_df = win_df.rename(columns={"Date":"Date_win"})
    
    # Strategy 1: Exact match on all three keys
    merged = tips.merge(win_df, on=["track_fp", "race_num", "horse_fp"], how="left")
    
    # Strategy 2: Fallback for unmatched tips
    unmatched_tips = merged[merged["bsp"].isna()].copy()
    
    if not unmatched_tips.empty:
        # Columns to keep for the fallback merge
        unmatched_tips_cols_to_keep = [col for col in unmatched_tips.columns if col not in win_df.columns or col in ["track_fp", "race_num", "horse_fp"]]
        
        fallback_merge = unmatched_tips[unmatched_tips_cols_to_keep].merge(
            win_df,
            on=["race_num", "horse_fp"],
            how="left"
        )
        
        # Ensure 'Date_win' is handled correctly
        if "Date_win" in fallback_merge.columns and "Date_tip" in fallback_merge.columns:
            fallback_merge["date_diff"] = (pd.to_datetime(fallback_merge["Date_win"]) - pd.to_datetime(fallback_merge["Date_tip"])).abs().dt.days
            fallback_merge = fallback_merge[fallback_merge["date_diff"] <= 2]
        
            # For tips with multiple fallback matches, pick the one with the smallest date difference
            idx = fallback_merge.groupby("tip_id")["date_diff"].idxmin()
            best_fallback = fallback_merge.loc[idx]
            
            # Replace NaN values in the original merge with the fallback matches
            merged.set_index("tip_id", inplace=True)
            merged.loc[best_fallback["tip_id"], win_df.columns] = best_fallback.set_index("tip_id")[win_df.columns]
            merged.reset_index(inplace=True)
    
    return merged
# ----------------------------
# Main Analysis
# ----------------------------
def perform_full_analysis(dataframes: Dict) -> Dict:
    response = {"daily_summary": [], "charts": {}}
    tips_df, race_data_df, win_prices_df = (dataframes.get(k) for k in ("tips","race_data","win_prices"))

    chart_keys = [
        "cumulative_profit","rolling_roi","roi_by_tipster","roi_by_odds",
        "price_movement_histogram","clv_trend","win_rate_vs_field_size"
    ]
    for k in chart_keys:
        response["charts"][k] = (
            {"labels": [], "datasets": []}
            if k in {"cumulative_profit","rolling_roi","clv_trend"}
            else {"labels": [], "data": []}
        )

    if tips_df is None or tips_df.empty:
        return response

    # Merge race data
    if race_data_df is not None and not race_data_df.empty:
        merged = tips_df.merge(
            race_data_df[["track_fp","race_num","horse_fp","BestOdds","field_size"]],
            on=["track_fp","race_num","horse_fp"], how="left"
        )
    else:
        merged = tips_df.copy()
        merged["BestOdds"] = np.nan
        merged["field_size"] = np.nan

    # Merge win prices
    if win_prices_df is not None and not win_prices_df.empty:
        merged = match_win_prices_to_tips(merged, win_prices_df)
    else:
        merged["bsp"] = merged["morningwap"] = merged["win_lose"] = merged["Date_win"] = np.nan

    merged["Date"] = pd.to_datetime(merged["Date_win"]).dt.date
    merged["Date"] = merged["Date"].fillna(pd.to_datetime(merged["Date"]).dt.date)
    for c in ["bsp","morningwap","BestOdds","win_lose","field_size"]:
        merged[c] = as_numeric_safe(merged.get(c, np.nan))
    merged["win_lose"] = merged["win_lose"].fillna(0)
    merged["Profit"] = np.where(merged["win_lose"]==1, merged["bsp"], 0) - 1
    merged.loc[merged["bsp"].isna(),"Profit"] = -1

    # ----- Daily Summary -----
    for d, g in merged.groupby("Date", dropna=False):
        bets = len(g)
        rtn  = g.loc[g["win_lose"]==1,"bsp"].fillna(0).sum()
        denom= max(bets,1)
        valid= (g["bsp"]>1)&(g["BestOdds"]>1)
        response["daily_summary"].append({
            "Date": str(d),
            "Bets Placed": bets,
            "Units Staked": bets,
            "Units Returned": float(rtn),
            "ROI %": (float(rtn - bets)/denom)*100,
            "Win Rate %": g["win_lose"].fillna(0).mean()*100,
            "Avg Odds": float(g["bsp"].dropna().mean()) if not pd.isna(g["bsp"].dropna().mean()) else 0,
            "CLV": float((((g.loc[valid,"bsp"]/g.loc[valid,"BestOdds"])-1).mean()*100) if valid.any() else 0),
            "Drifters %": float(((g["bsp"]>g["morningwap"]).sum()/max((g["morningwap"]>1).sum(),1))*100 if "morningwap" in g.columns else 0),
            "Steamers %": float(((g["bsp"]<g["morningwap"]).sum()/max((g["morningwap"]>1).sum(),1))*100 if "morningwap" in g.columns else 0),
        })

    merged["Date"]=pd.to_datetime(merged["Date"]).dt.date

    # ----- 1) Cumulative Profit -----
    dp = merged.groupby(["Tip Website","Date"], as_index=False)["Profit"].sum().sort_values(["Tip Website","Date"])
    pivot = dp.pivot(index="Date", columns="Tip Website", values="Profit").fillna(0).cumsum()
    response["charts"]["cumulative_profit"] = {
        "labels": [d.strftime("%Y-%m-%d") for d in pivot.index],
        "datasets": [{"name": c, "data": pivot[c].round(4).tolist()} for c in pivot.columns]
    }

    # ----- 2) 30-Day Rolling ROI -----
    ds = merged.groupby(["Tip Website","Date"]).agg(
        bets=("Profit","size"),
        rtn=("bsp",lambda s:s.fillna(0).sum())
    ).reset_index()
    ds["roi"] = (ds["rtn"] - ds["bets"]) / ds["bets"].replace(0, np.nan)

    out = []
    for tip, g in ds.groupby("Tip Website"):
        g = g.sort_values("Date").set_index("Date")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(idx)
        g["roi"] = g["roi"].fillna(0)
        g["roi30"] = g["roi"].rolling(30, min_periods=1).mean()
        g["Tip Website"] = tip
        g["Date"] = g.index.date
        out.append(g.reset_index(drop=True)[["Tip Website","Date","roi30"]])

    if out:
        rr = pd.concat(out)
        p = rr.pivot(index="Date", columns="Tip Website", values="roi30").ffill().fillna(0)
        response["charts"]["rolling_roi"] = {
            "labels": [d.strftime("%Y-%m-%d") for d in p.index],
            "datasets": [{"name": c, "data": (p[c] * 100).round(4).tolist()} for c in p.columns]
        }

    # ----- 3) ROI by tipster -----
    rbt = merged.groupby("Tip Website")["Profit"].mean().fillna(0) * 100
    response["charts"]["roi_by_tipster"] = {
        "labels": rbt.index.tolist(),
        "data"  : rbt.round(4).tolist()
    }

    # ----- 4) ROI by odds band -----
    merged["odds_bin"] = pd.cut(
        merged["bsp"].where(merged["bsp"] > 1),
        bins=[1, 3, 5, 10, 20, 50, 1000],
        labels=["$1-3", "$3-5", "$5-10", "$10-20", "$20-50", "$50+"],
        include_lowest=True
    ).astype(str)

    rob = merged.groupby("odds_bin")["Profit"].mean().fillna(0) * 100
    response["charts"]["roi_by_odds"] = {
        "labels": rob.index.tolist(),
        "data": rob.round(4).tolist()
    }

    # ----- 5) Price movement histogram -----
    pm = (merged["bsp"] > 1) & (merged["morningwap"] > 1)
    pmv = ((merged.loc[pm,"bsp"] - merged.loc[pm,"morningwap"]) / merged.loc[pm,"morningwap"]).astype(float)
    response["charts"]["price_movement_histogram"] = {
        "data": pmv.round(6).tolist()
    }

    # ----- 6) CLV Trend -----
    valid = (merged["bsp"]>1)&(merged["BestOdds"]>1)
    merged["clv_calc"] = ((merged["bsp"] / merged["BestOdds"]) - 1) * 100
    clvt = merged.loc[valid].groupby("Date")["clv_calc"].mean()
    response["charts"]["clv_trend"] = {
        "labels":[d.strftime("%Y-%m-%d") for d in clvt.index],
        "datasets":[{"name":"CLV","data":clvt.round(4).tolist()}]
    }

    # ----- 7) Win rate vs field size -----
    if "field_size" in merged.columns:
        wr = merged.dropna(subset=["field_size"]).copy()
        wr["field_size"] = wr["field_size"].round(0).astype(int)
        grp = wr.groupby("field_size")["win_lose"].mean().fillna(0) * 100
        response["charts"]["win_rate_vs_field_size"] = {
            "labels": [str(i) for i in grp.index],
            "data": grp.round(4).tolist()
        }

    return clean_for_json(response)


# ----------------------------
# POST /analyze/
# ----------------------------
@app.post("/analyze/")
async def analyze_betting_files(files: List[UploadFile] = File(...)):
  if len(files) < 3:
    raise HTTPException(status_code=400, detail="Please upload tips, race_data, and win_prices files.")
  file_map={"tips":None,"race_data":None,"win_prices":None}

  for file in files:
    content = await file.read()
    try: df = pd.read_csv(io.BytesIO(content))
    except:
      try: df = pd.read_excel(io.BytesIO(content))
      except: raise HTTPException(status_code=400, detail=f"Could not read file: {file.filename}")

    cols = {c.strip() for c in df.columns}
    if {"Tip Website","Track"}.issubset(cols) or "First Selection Name" in cols:
      file_map["tips"] = df
    elif {"HorseName","RaceTrack"}.issubset(cols) or {"RaceTrack","RaceNum"}.issubset(cols):
      file_map["race_data"] = df
    elif "bsp" in cols and "win_lose" in cols:
      if "event_name" not in df.columns or not df["event_name"].astype(str).str.contains("To Be Placed",na=False).any():
        file_map["win_prices"] = df

  if any(v is None for v in file_map.values()):
    raise HTTPException(status_code=400, detail="One or more required file types could not be identified.")
  try:
    cleaned = clean_and_standardize_data(file_map)
    return perform_full_analysis(cleaned)
  except HTTPException: raise
  except Exception as e:
    tb = traceback.format_exc()
    logger.error(f"Analysis error: {e}\n{tb}")
    raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")


# ----------------------------
# Health & static frontend
# ----------------------------
@app.get("/health")
async def health(): return {"status": "healthy"}

script_dir = os.path.dirname(__file__)
frontend_dir = os.path.join(os.path.dirname(script_dir), "frontend")
if os.path.exists(frontend_dir):
  @app.get("/favicon.ico", include_in_schema=False)
  async def favicon(): return FileResponse(os.path.join(frontend_dir, 'favicon_io', 'favicon.ico'))
  app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
  @app.get("/", response_class=FileResponse)
  async def root(): return FileResponse(os.path.join(frontend_dir, 'index.html'))


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8000)

