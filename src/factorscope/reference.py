from __future__ import annotations

import io
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_DATE_RE = re.compile(r"^\s*(\d{8})\s*,")

_FILES = {
    "ff3": ("F-F_Research_Data_Factors_daily_CSV.zip", ["Mkt-RF", "SMB", "HML", "RF"]),
    "ff5": ("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
            ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]),
    "mom": ("F-F_Momentum_Factor_daily_CSV.zip", ["Mom"]),
}

_PORTFOLIOS = {
    "25_size_bm": ("25_Portfolios_5x5_daily_CSV.zip", 25),
    "100_size_bm": ("100_Portfolios_10x10_daily_CSV.zip", 100),
    "49_industry": ("49_Industry_Portfolios_daily_CSV.zip", 49),
    "5_industry": ("5_Industry_Portfolios_daily_CSV.zip", 5),
}


def _fetch_zip_csv(fname, timeout=60):
    with urllib.request.urlopen(_BASE + fname, timeout=timeout) as r:
        raw = r.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def _parse_daily(text, ncols):
    dates, rows, started = [], [], False
    for line in text.splitlines():
        m = _DATE_RE.match(line)
        if not m:
            if started:
                break
            continue
        vals = [p.strip() for p in line.split(",")][1:]
        if len(vals) != ncols:
            if started:
                break
            continue
        try:
            fv = [float(v) for v in vals]
        except ValueError:
            if started:
                break
            continue
        started = True
        dates.append(pd.to_datetime(m.group(1), format="%Y%m%d"))
        rows.append(fv)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    return df.replace([-99.99, -999.0], np.nan)


def load_reference_factors(kind="ff5", start=None, end=None, include_momentum=True):
    fname, cols = _FILES[kind]
    df = _parse_daily(_fetch_zip_csv(fname), len(cols))
    df.columns = cols
    df = df.drop(columns=["RF"], errors="ignore")
    if include_momentum:
        try:
            mom = _parse_daily(_fetch_zip_csv(_FILES["mom"][0]), 1)
            mom.columns = ["Mom"]
            df = df.join(mom, how="left")
        except Exception:
            pass
    df = df.dropna(how="all")
    if start is not None:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df.index <= pd.to_datetime(end)]
    return df


def load_portfolios(kind="100_size_bm", start=None, end=None):
    if kind not in _PORTFOLIOS:
        raise ValueError(f"kind must be one of {list(_PORTFOLIOS)}")
    fname, ncols = _PORTFOLIOS[kind]
    df = _parse_daily(_fetch_zip_csv(fname), ncols)
    df.columns = [f"P{i+1:03d}" for i in range(ncols)]
    df = df.dropna(how="all")
    if start is not None:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df.loc[df.index <= pd.to_datetime(end)]
    return df
