# -*- coding: utf-8 -*-
"""
01_fetch_sources.py - download all authoritative data sources into data/raw/

Sources (verified):
  1. UN WPP 2024 (population/TFR/urban)   population.un.org direct CSV.gz link
  2. OWID per-capita electricity          ourworldindata.org/grapher/per-capita-electricity-generation.csv
  3. OWID steel production                ourworldindata.org/grapher/steel-production.csv (validated at runtime)
  4. World Bank WDI API                   api.worldbank.org (12 indicators, 1960:2024)
  5. Barro-Lee education                  barrolee.com (female 15+ mean years of schooling)
  6. WBL maternity/paternity leave panel  wbl.worldbank.org
  7. WHO GHO health workforce             ghoapi.azureedge.us/api/
  8. Energy Institute refining (best-effort)  energyinst.org/statistical-review
  9. OECD family/housing database (best-effort)  oecd.org
  10. IRF road statistics / UNECE rail (best-effort)

Idempotent: existing files are skipped. Status of every source is recorded in
data/raw/source_log.json
Usage: python 01_fetch_sources.py [--force]
"""
import gzip
import io
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

LOG = os.path.join(RAW, "source_log.json")
FORCE = "--force" in sys.argv

UA = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")}

# The machine's system proxy points to an unresponsive 127.0.0.1:7890; bypass it
# and connect directly.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(_opener)


def _fetch(url, timeout=120, binary=False, max_retry=3):
    """Direct download with retries; returns bytes; raises RuntimeError on failure"""
    last = None
    for attempt in range(max_retry):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) < 100 and not binary:
                raise RuntimeError("Response too small (likely a 404 page)")
            return data
        except Exception as e:
            last = e
            if attempt < max_retry - 1:
                time.sleep(2 + 2 * attempt)
    raise RuntimeError(f"Download failed: {url} ({last})")


def save(name, data, binary=False):
    """Write into raw/; skip if the file exists and --force is not given"""
    path = os.path.join(RAW, name)
    if not FORCE and os.path.exists(path) and os.path.getsize(path) > 0:
        return path, "skip(exists)"
    mode = "wb" if binary else "w"
    if binary:
        with open(path, mode) as f:
            f.write(data)
    else:
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        with open(path, mode, encoding="utf-8") as f:
            f.write(data)
    return path, "ok"


STATUS = {}


def log_source(name, url, status, note=""):
    STATUS[name] = {"url": url, "status": status, "note": note,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump(STATUS, f, ensure_ascii=False, indent=1)
    print(f"[{status.upper():6}] {name}: {note}")


# ============ 1. UN WPP 2024 ============
WPP_URL = ("https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)"
           "/CSV_FILES/WPP2024_Demographic_Indicators_Medium.csv.gz")


def fetch_wpp():
    """WPP is a large file (rate-limited ~13KB/s); supports external curl resume:
    skips if the .gz already exists"""
    url = WPP_URL
    gz_path = os.path.join(RAW, "wpp2024_demo.csv.gz")
    out = os.path.join(RAW, "wpp2024_demo.csv")
    if not os.path.exists(gz_path) or os.path.getsize(gz_path) == 0:
        save("wpp2024_demo.csv.gz", _fetch(url, binary=True), binary=True)
        log_source("wpp", url, "ok", f"{os.path.getsize(gz_path)} bytes")
    else:
        log_source("wpp", url, "ok", f"cached (possibly mid-resume) {os.path.getsize(gz_path)} bytes")
    # Decompress to plain CSV (tolerate an incomplete .gz)
    if not os.path.exists(out):
        try:
            with gzip.open(gz_path, "rb") as f:
                data = f.read()
            with open(out, "wb") as f:
                f.write(data)
            log_source("wpp_unzip", url, "ok", f"{len(data)} bytes -> {out}")
        except Exception as e:
            log_source("wpp_unzip", url, "pending", f"gz incomplete (still downloading): {str(e)[:80]}")


# ============ 2/3. OWID ============
def fetch_owid(slug, name):
    url = f"https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full"
    try:
        data = _fetch(url)
        if b"not found" in data[:2000].lower():
            raise RuntimeError("404: slug does not exist")
        path, st = save(name, data)
        log_source(name, url, "ok" if st == "ok" else "skip",
                   f"{len(data)} bytes")
    except Exception as e:
        log_source(name, url, "fail", str(e)[:200])


# ============ 4. World Bank WDI ============
WDI_INDICATORS = {
    "gdppc_kd":        "NY.GDP.PCAP.KD",      # GDP per capita, 2015 constant USD
    "elec_access":     "EG.ELC.ACCS.ZS",      # access to electricity, % of population
    "urban":           "SP.URB.TOTL.IN.ZS",   # urbanization rate, %
    "mobile":          "IT.CEL.SETS.P2",      # mobile cellular subscriptions per 100 people
    "broadband":       "IT.NET.BBND.P2",      # fixed broadband subscriptions per 100 people
    "manuf_kd":        "NV.IND.MANF.KD",      # manufacturing value added, 2015 constant USD (total)
    "physicians":      "SH.MED.PHYS.ZS",      # physicians per 1000 people
    "literacy":        "SE.ADT.LITR.ZS",      # adult literacy rate, %
    "agri_empl":       "SL.AGR.EMPL.ZS",      # employment in agriculture, % of total employment
    "pop":             "SP.POP.TOTL",         # total population (backup)
    "us_gdp_defl":     "NY.GDP.DEFL.ZS",      # GDP deflator (USA only, for USD conversion)
}


def fetch_wb_all():
    for name, ind in WDI_INDICATORS.items():
        url = (f"https://api.worldbank.org/v2/country/ALL/indicator/{ind}"
               f"?format=json&per_page=10000&date=1960:2024")
        out = os.path.join(RAW, f"wdi_{name}.csv")
        if not FORCE and os.path.exists(out) and os.path.getsize(out) > 0:
            log_source(f"wdi_{name}", url, "ok", "cached")
            continue
        try:
            rows = []
            page = 1
            pages = 1
            while page <= pages:
                u = url + f"&page={page}"
                d = json.loads(_fetch(u))
                if not isinstance(d, list) or len(d) < 2 or not d[1]:
                    break
                for e in d[1]:
                    if e.get("value") is not None:
                        iso = (e.get("countryiso3code") or "").strip()
                        if iso:
                            rows.append(f"{iso},{e['date']},{e['value']}")
                pages = d[0]["pages"]
                page += 1
                time.sleep(0.3)
            with open(out, "w", encoding="utf-8") as f:
                f.write("iso3,year,value\n" + "\n".join(rows) + "\n")
            log_source(f"wdi_{name}", url, "ok", f"{len(rows)} rows")
        except Exception as e:
            log_source(f"wdi_{name}", url, "fail", str(e)[:200])


# ============ 5. Barro-Lee ============
def fetch_barro_lee():
    """Female 15+ mean years of schooling, 1950-2015 every 5 years. Scrape the
    official data page for links, with a GitHub mirror as fallback"""
    url = "http://barrolee.com/"
    out = os.path.join(RAW, "barro_lee.csv")
    if not FORCE and os.path.exists(out) and os.path.getsize(out) > 0:
        log_source("barro_lee", url, "ok", "cached")
        return
    try:
        html = _fetch(url, timeout=60).decode("utf-8", errors="replace")
        links = re.findall(r'href=["\']([^"\']+?\.(?:xlsx?|csv|dta))["\']', html, re.I)
        # Prefer the female 15+ schooling file
        cand = [l for l in links if re.search(r"(F|female|yr_sch|YRS_SCH|yrsch)", l, re.I)]
        cand = cand or links
        chosen, data = None, None
        for l in cand:
            u = l if l.startswith("http") else "http://barrolee.com/" + l.lstrip("./")
            try:
                data = _fetch(u, timeout=120, binary=True)
                chosen = u
                break
            except Exception:
                continue
        if data is None:
            raise RuntimeError("No downloadable data file on the official site")
        ext = os.path.splitext(chosen)[1].lower()
        if ext == ".csv":
            save("barro_lee.csv", data)
        else:  # xlsx/dta kept as-is, read with pandas in stage 02
            save("barro_lee" + ext, data, binary=True)
        log_source("barro_lee", url, "ok", f"{chosen}")
    except Exception as e:
        # Official GitHub mirror: female 15+ mean years of schooling, 1950-2015 every 5 years
        gh = ("https://raw.githubusercontent.com/barrolee/BarroLeeDataSet/"
              "master/BLData/BL_v3_F.csv")
        try:
            data = _fetch(gh, timeout=120)
            save("barro_lee.csv", data)
            log_source("barro_lee", url, "ok", f"GitHub mirror {gh}")
        except Exception as e2:
            log_source("barro_lee", url, "fail", f"official: {str(e)[:100]}; GitHub: {str(e2)[:100]}")


# ============ 6. WBL maternity/paternity leave ============
WBL_CANDIDATES = [
    "https://wbl.worldbank.org/content/dam/sites/wbl/documents/2024/WBL2024-1-0-Historical-Panel-Data.xlsx",
    "https://wbl.worldbank.org/content/dam/sites/wbl/documents/2024/WBL2024-2-0.xlsx",
]


def fetch_wbl():
    url = "https://wbl.worldbank.org/en/wbl-data"
    for i, c in enumerate(WBL_CANDIDATES):
        try:
            data = _fetch(c, timeout=120, binary=True)
            save("wbl_panel.xlsx", data, binary=True)
            log_source("wbl", url, "ok", c)
            return
        except Exception as e:
            last = str(e)[:120]
    # Scrape the data page for links
    try:
        html = _fetch(url, timeout=90).decode("utf-8", errors="replace")
        links = re.findall(r'href=["\']([^"\']+\.(?:xlsx?|csv))["\']', html, re.I)
        for l in links:
            try:
                data = _fetch(l if l.startswith("http") else "https:" + l, timeout=120, binary=True)
                save("wbl_panel.xlsx", data, binary=True)
                log_source("wbl", url, "ok", l)
                return
            except Exception:
                continue
    except Exception as e:
        last = str(e)[:120]
    log_source("wbl", url, "fail", last)


# ============ 7. WHO GHO health workforce ============
def fetch_gho(indicator, name, label):
    url = f"https://ghoapi.azureedge.us/api/{indicator}?$select=SpatialDim,TimeDim,Value&$filter=TimeDim%20ge%201990"
    out = os.path.join(RAW, f"gho_{name}.csv")
    if not FORCE and os.path.exists(out) and os.path.getsize(out) > 0:
        log_source(f"gho_{name}", url, "ok", "cached")
        return
    try:
        rows = []
        skip = 0
        while True:
            u = f"{url}&$skip={skip}&$top=1000"
            d = json.loads(_fetch(u))
            recs = d.get("value", [])
            for r in recs:
                if r.get("Value") is not None and r.get("SpatialDim"):
                    rows.append(f"{r['SpatialDim']},{r['TimeDim']},{r['Value']}")
            if len(recs) < 1000:
                break
            skip += 1000
            time.sleep(0.2)
        with open(out, "w", encoding="utf-8") as f:
            f.write("iso3,year,value\n" + "\n".join(rows) + "\n")
        log_source(f"gho_{name}", url, "ok", f"{len(rows)} rows ({label})")
    except Exception as e:
        log_source(f"gho_{name}", url, "fail", str(e)[:200])


# ============ 8/9. Energy Institute / OECD (best-effort) ============
def fetch_energy_inst():
    url = "https://www.energyinst.org/statistical-review"
    try:
        html = _fetch(url, timeout=90).decode("utf-8", errors="replace")
        links = re.findall(r'href=["\']([^"\']+\.xlsx?)["\']', html, re.I)
        if not links:
            log_source("energy_inst", url, "fail", "no xlsx direct links (requires interactive download)")
            return
        # Prefer refining-related files
        cand = [l for l in links if re.search(r"refin", l, re.I)] or links
        u = cand[0] if cand[0].startswith("http") else "https://www.energyinst.org" + cand[0].lstrip("/")
        data = _fetch(u, timeout=180, binary=True)
        save("energy_inst_refining.xlsx", data, binary=True)
        log_source("energy_inst", url, "ok", u)
    except Exception as e:
        log_source("energy_inst", url, "fail", str(e)[:200])


def fetch_oecd():
    url = "https://www.oecd.org/en/data/indicators/net-childcare-costs.html"
    try:
        html = _fetch(url, timeout=90).decode("utf-8", errors="replace")
        links = re.findall(r'href=["\']([^"\']+\.(?:xlsx?|csv))["\']', html, re.I)
        log_source("oecd", url, "ok" if links else "fail",
                   f"{len(links)} xlsx/csv links" if links else "no direct links on page")
    except Exception as e:
        log_source("oecd", url, "fail", str(e)[:200])


def fetch_irf_unece():
    url = "https://worldroadstatistics.org/get-data/"
    try:
        html = _fetch(url, timeout=90).decode("utf-8", errors="replace")
        has_link = bool(re.search(r'(download|api|\.csv|\.xlsx)', html, re.I))
        log_source("irf", url, "ok" if has_link else "fail",
                   "download entry exists" if has_link else "paywall/registration required (no programmatic access to historical data)")
    except Exception as e:
        log_source("irf", url, "fail", str(e)[:200])


def main():
    print("== Downloading all data sources ==")
    fetch_wpp()
    fetch_owid("per-capita-electricity-generation", "owid_elec_gen_pc.csv")
    fetch_owid("steel-production", "owid_steel.csv")
    fetch_wb_all()
    fetch_barro_lee()
    fetch_wbl()
    fetch_gho("HRH_0001", "physicians", "physician density per 10000")
    fetch_gho("HRH_0006", "nurses", "nursing/midwifery density per 10000")
    fetch_energy_inst()
    fetch_oecd()
    fetch_irf_unece()
    print("\n== Done ==")
    for k, v in STATUS.items():
        print(f"  {v['status']:6} {k}")


if __name__ == "__main__":
    main()
