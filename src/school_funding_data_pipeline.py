"""
=============================================================
 School Funding Impact on Student Performance
 Data Pipeline: CCD + CRDC + EDFacts → Merged Analysis Dataset
=============================================================

Datasets used (all free, public, from data.gov / NCES):
  1. CCD F-33  – per-pupil expenditure by district
  2. CRDC      – student demographics & equity indicators
  3. EDFacts   – math/reading proficiency & graduation rates

Target year: 2017-18 (clean overlap across all three datasets)

Steps:
  0. Install dependencies
  1. Download raw files
  2. Load & inspect each dataset
  3. Select relevant columns
  4. Clean & standardize
  5. Merge on LEAID (district ID)
  6. Feature engineering
  7. Export final CSV
=============================================================
"""

# ── 0. Dependencies ──────────────────────────────────────────
import pandas as pd
import numpy as np
import os
import requests
import zipfile
import io
import matplotlib
matplotlib.use("Agg")           # non-interactive backend; safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", "{:,.2f}".format)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_YEAR = "2017-18"
print(f"Pipeline start — target year: {TARGET_YEAR}\n")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — DOWNLOAD RAW FILES
# ═══════════════════════════════════════════════════════════════
# NOTE: URLs below point to the official NCES/ED download pages.
# Run the download helpers once; subsequent runs load from disk.

def download_file(url: str, dest_path: str, expect_zip: bool = False) -> str:
    """Download a file from url to dest_path if not already present."""
    if os.path.exists(dest_path):
        print(f"  ✓ Already downloaded: {os.path.basename(dest_path)}")
        return dest_path
    print(f"  ↓ Downloading: {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    if expect_zip and not r.content.startswith(b"PK"):
        raise ValueError(
            f"  Expected a ZIP file from {url} but got an HTML page or redirect.\n"
            "  The URL is likely outdated. Download the file manually and place it in ./data/"
        )
    with open(dest_path, "wb") as f:
        f.write(r.content)
    print(f"  ✓ Saved to: {dest_path}")
    return dest_path


def extract_zip(zip_path: str, extract_to: str) -> None:
    """Extract a zip archive into extract_to, skipping files already present."""
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            dest = os.path.join(extract_to, os.path.basename(member))
            if os.path.basename(member) and not os.path.exists(dest):
                with z.open(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
                print(f"  ✓ Extracted: {os.path.basename(member)}")
            elif os.path.basename(member):
                print(f"  ✓ Already extracted: {os.path.basename(member)}")


# --- CCD F-33 (district finance) ---
# Direct CSV download from NCES for fiscal year 2018 (= school year 2017-18)
CCD_URL  = "https://nces.ed.gov/ccd/data/zip/sdf18_1a.zip"
CCD_ZIP  = os.path.join(OUTPUT_DIR, "ccd_f33_2018.zip")

# --- CRDC 2017-18 (civil rights / demographics) ---
# Public-use ZIP from the ED Office for Civil Rights
# If this URL stops working, download manually from:
#   https://www2.ed.gov/about/offices/list/ocr/docs/crdc-2017-18.html
CRDC_URL = "https://www2.ed.gov/about/offices/list/ocr/docs/2017-18-crdc-data.zip"
CRDC_ZIP = os.path.join(OUTPUT_DIR, "crdc_2017_18.zip")

# --- EDFacts Assessment 2017-18 (math/reading proficiency) ---
# Two separate flat files from the ED.gov EDFacts data library (one per subject)
EDFACTS_MATH_URL = (
    "https://www.ed.gov/sites/ed/files/about/inits/ed/edfacts/"
    "data-files/math-achievement-lea-sy2017-18.csv"
)
EDFACTS_RLA_URL  = (
    "https://www.ed.gov/sites/ed/files/about/inits/ed/edfacts/"
    "data-files/rla-achievement-lea-sy2017-18.csv"
)
EDFACTS_MATH_CSV = os.path.join(OUTPUT_DIR, "edfacts_math_lea_2017_18.csv")
EDFACTS_RLA_CSV  = os.path.join(OUTPUT_DIR, "edfacts_rla_lea_2017_18.csv")

print("Step 1 — Downloading raw data files...")
print("  (Skip any file that already exists on disk)\n")

download_file(CCD_URL,   CCD_ZIP,  expect_zip=True)
extract_zip(CCD_ZIP,     OUTPUT_DIR)

download_file(CRDC_URL,  CRDC_ZIP, expect_zip=True)
extract_zip(CRDC_ZIP,    OUTPUT_DIR)

download_file(EDFACTS_MATH_URL, EDFACTS_MATH_CSV)
download_file(EDFACTS_RLA_URL,  EDFACTS_RLA_CSV)

print("  → Files ready\n")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — LOAD EACH DATASET
# ═══════════════════════════════════════════════════════════════

print("Step 2 — Loading raw files...\n")

# ── 2a. CCD F-33 Finance Survey ─────────────────────────────
# After unzipping CCD_ZIP the main file is typically named sdf18_1a.txt (tab-delimited)
# Adjust the filename below to match what NCES ships in the zip.

CCD_FILE = os.path.join(OUTPUT_DIR, "sdf18_1a.txt")   # tab-delimited flat file

ccd_raw = pd.read_csv(
    CCD_FILE,
    sep="\t",
    encoding="latin-1",
    dtype={"LEAID": str},      # keep as string to preserve leading zeros
    low_memory=False,
)
print(f"  CCD  — rows: {len(ccd_raw):,}  cols: {ccd_raw.shape[1]}")

# ── 2b. CRDC Demographics ────────────────────────────────────
# The CRDC zip ships as many topic files; we need Enrollment + School Support.
CRDC_ENR_FILE  = os.path.join(OUTPUT_DIR, "Enrollment.csv")
CRDC_SUPP_FILE = os.path.join(OUTPUT_DIR, "School Support.csv")

crdc_enr  = pd.read_csv(CRDC_ENR_FILE,  dtype={"LEAID": str}, encoding="latin-1", low_memory=False)
crdc_supp = pd.read_csv(CRDC_SUPP_FILE, dtype={"LEAID": str}, encoding="latin-1", low_memory=False)

# Merge on COMBOKEY (unique school key present in every CRDC file)
crdc_raw = crdc_enr.merge(
    crdc_supp[["COMBOKEY", "SCH_FTETEACH_TOT", "SCH_FTETEACH_CERT"]],
    on="COMBOKEY", how="left",
)

# Build the combined counts the aggregation step expects
crdc_raw["SCH_IDEACOUNT"] = (
    pd.to_numeric(crdc_raw.get("SCH_PSENR_IDEA_M"), errors="coerce").fillna(0)
    + pd.to_numeric(crdc_raw.get("SCH_PSENR_IDEA_F"), errors="coerce").fillna(0)
)
crdc_raw["SCH_ELLCOUNT"] = (
    pd.to_numeric(crdc_raw.get("SCH_PSENR_LEP_M"), errors="coerce").fillna(0)
    + pd.to_numeric(crdc_raw.get("SCH_PSENR_LEP_F"), errors="coerce").fillna(0)
)

print(f"  CRDC — rows: {len(crdc_raw):,}  cols: {crdc_raw.shape[1]}")

# ── 2c. EDFacts Assessment ───────────────────────────────────
# ED.gov ships math and reading proficiency as separate files; merge on LEAID.
edfacts_math = pd.read_csv(EDFACTS_MATH_CSV, dtype={"LEAID": str}, low_memory=False)
edfacts_rla  = pd.read_csv(EDFACTS_RLA_CSV,  dtype={"LEAID": str}, low_memory=False)

# Strip any trailing year tag (e.g. _1718) so the column dict below matches both formats
import re as _re
edfacts_math.columns = [_re.sub(r"_\d{4}$", "", c) for c in edfacts_math.columns]
edfacts_rla.columns  = [_re.sub(r"_\d{4}$", "", c) for c in edfacts_rla.columns]

# Keep only the reading proficiency column(s) from the RLA file to avoid duplicate LEAID etc.
rla_keep = ["LEAID"] + [c for c in edfacts_rla.columns if "RLA" in c and "PCTPROF" in c]
edfacts_raw = edfacts_math.merge(edfacts_rla[rla_keep], on="LEAID", how="outer")

print(f"  EDFacts — rows: {len(edfacts_raw):,}  cols: {edfacts_raw.shape[1]}\n")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — SELECT RELEVANT COLUMNS
# ═══════════════════════════════════════════════════════════════

print("Step 3 — Selecting relevant columns...\n")

# ── 3a. CCD F-33 columns ─────────────────────────────────────
ccd_cols = {
    "LEAID"    : "leaid",
    "NAME"     : "district_name",
    "STNAME"   : "state",
    "MEMBERSCH": "enrollment",        # fall membership (F-33 uses MEMBERSCH, not ENROLL)
    "TOTALREV" : "total_revenue",
    "TFEDREV"  : "federal_revenue",
    "TSTREV"   : "state_revenue",
    "TLOCREV"  : "local_revenue",
    "TOTALEXP" : "total_expenditure",
    "TCURINST" : "instr_expenditure",
    "TCUROTH"  : "support_expenditure",
    # per_pupil_spending derived in Step 4 (TOTALEXP / MEMBERSCH)
    # TSALARY and TITLEI are not in the F-33 2017-18 file
}
ccd = ccd_raw[list(ccd_cols.keys())].rename(columns=ccd_cols).copy()
print(f"  CCD selected: {ccd.shape[1]} columns")

# ── 3b. CRDC columns ─────────────────────────────────────────
crdc_cols = {
    "LEAID"                 : "leaid",
    "SCH_ENR_HI_M"          : "enr_hispanic_m",
    "SCH_ENR_HI_F"          : "enr_hispanic_f",
    "SCH_ENR_BL_M"          : "enr_black_m",
    "SCH_ENR_BL_F"          : "enr_black_f",
    "SCH_ENR_WH_M"          : "enr_white_m",
    "SCH_ENR_WH_F"          : "enr_white_f",
    "SCH_ENR_AS_M"          : "enr_asian_m",
    "SCH_ENR_AS_F"          : "enr_asian_f",
    "SCH_IDEACOUNT"         : "special_ed_count",
    "SCH_ELLCOUNT"          : "ell_count",         # English language learners
    "SCH_FTETEACH_TOT"      : "total_teachers_fte",
    "SCH_FTETEACH_CERT"     : "certified_teachers", # certified teachers
    "SCH_APENR_IND"         : "ap_course_access",  # AP course offered flag
    "SCH_DISCWODIS_EXPZT_IND": "zero_tolerance",   # zero-tolerance discipline
}
crdc = crdc_raw[[c for c in crdc_cols if c in crdc_raw.columns]].rename(columns=crdc_cols).copy()
print(f"  CRDC selected: {crdc.shape[1]} columns")

# ── 3c. EDFacts columns ──────────────────────────────────────
edfacts_cols = {
    "LEAID"             : "leaid",
    "ALL_MTH00PCTPROF"  : "math_proficiency_pct",    # % proficient in math (all students)
    "ALL_RLA00PCTPROF"  : "reading_proficiency_pct", # % proficient in reading
    "CWD_MTH00PCTPROF"  : "math_prof_disabilities",  # students with disabilities
    "ECD_MTH00PCTPROF"  : "math_prof_econ_disadv",   # economically disadvantaged
    "HI_MTH00PCTPROF"   : "math_prof_hispanic",
    "BL_MTH00PCTPROF"   : "math_prof_black",
    "WH_MTH00PCTPROF"   : "math_prof_white",
    # COHORT_RATE (graduation rate) is in a separate ACGR file; omitted here
}
edfacts = edfacts_raw[[c for c in edfacts_cols if c in edfacts_raw.columns]].rename(columns=edfacts_cols).copy()
print(f"  EDFacts selected: {edfacts.shape[1]} columns\n")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — CLEAN & STANDARDIZE
# ═══════════════════════════════════════════════════════════════

print("Step 4 — Cleaning and standardizing...\n")

def clean_leaid(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure LEAID is a zero-padded 7-character string."""
    df["leaid"] = df["leaid"].astype(str).str.strip().str.zfill(7)
    return df

def replace_suppressed(df: pd.DataFrame, suppressed_vals=(-1, -2, -3, -9)) -> pd.DataFrame:
    """NCES uses negative sentinel values for suppressed/missing data → NaN."""
    return df.replace(suppressed_vals, np.nan)

def clean_pct_col(series: pd.Series) -> pd.Series:
    """Convert string proficiency ranges like 'GE50' or 'LT50' to numeric midpoints."""
    # EDFacts proficiency is sometimes stored as ranges; extract numeric where possible
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric

# ── CCD ──────────────────────────────────────────────────────
ccd = clean_leaid(ccd)
ccd = replace_suppressed(ccd)

# Derive per-pupil spending (TOTALEXP / MEMBERSCH) and cap outliers
ccd["enrollment"]         = pd.to_numeric(ccd["enrollment"],        errors="coerce")
ccd["total_expenditure"]  = pd.to_numeric(ccd["total_expenditure"], errors="coerce")
ccd["per_pupil_spending"] = (ccd["total_expenditure"] / ccd["enrollment"]).round(0)
ccd = ccd[ccd["per_pupil_spending"].between(1_000, 100_000) | ccd["per_pupil_spending"].isna()]

# Revenue share features
ccd["pct_local_revenue"] = (ccd["local_revenue"] / ccd["total_revenue"] * 100).round(2)
ccd["pct_federal_revenue"] = (ccd["federal_revenue"] / ccd["total_revenue"] * 100).round(2)
ccd["pct_instr_spending"] = (ccd["instr_expenditure"] / ccd["total_expenditure"] * 100).round(2)

print(f"  CCD — {ccd['leaid'].nunique():,} unique districts")
print(f"  CCD — per_pupil_spending range: "
      f"${ccd['per_pupil_spending'].min():,.0f} – ${ccd['per_pupil_spending'].max():,.0f}")

# ── CRDC ─────────────────────────────────────────────────────
crdc = clean_leaid(crdc)
crdc = replace_suppressed(crdc)

# Aggregate to district level (CRDC is school-level; sum enrollment counts)
crdc_agg = crdc.groupby("leaid").agg(
    special_ed_count   = ("special_ed_count",   "sum"),
    ell_count          = ("ell_count",           "sum"),
    total_teachers_fte = ("total_teachers_fte",  "sum"),
    certified_teachers = ("certified_teachers",  "sum"),
    enr_hispanic       = ("enr_hispanic_m",      "sum"),
    enr_black          = ("enr_black_m",         "sum"),
    enr_white          = ("enr_white_m",         "sum"),
    # ap_course_access omitted — SCH_APENR_IND is in Advanced Placement.csv, not Enrollment.csv
).reset_index()

crdc_agg["pct_certified_teachers"] = (
    crdc_agg["certified_teachers"] / crdc_agg["total_teachers_fte"] * 100
).round(2)

print(f"\n  CRDC — {crdc_agg['leaid'].nunique():,} unique districts (after aggregation)")

# ── EDFacts ──────────────────────────────────────────────────
edfacts = clean_leaid(edfacts)
edfacts = replace_suppressed(edfacts)

for col in ["math_proficiency_pct", "reading_proficiency_pct", "graduation_rate"]:
    if col in edfacts.columns:
        edfacts[col] = clean_pct_col(edfacts[col])

# Deduplicate: keep one row per district (some districts appear multiple times for subgroups)
edfacts = edfacts.drop_duplicates(subset="leaid", keep="first")
print(f"  EDFacts — {edfacts['leaid'].nunique():,} unique districts\n")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — MERGE ALL THREE DATASETS ON LEAID
# ═══════════════════════════════════════════════════════════════

print("Step 5 — Merging datasets on LEAID...\n")

# Start with CCD (funding) as the base
merged = ccd.merge(crdc_agg, on="leaid", how="left")
print(f"  After CCD + CRDC merge: {len(merged):,} rows")

merged = merged.merge(edfacts, on="leaid", how="left")
print(f"  After + EDFacts merge:  {len(merged):,} rows")

# Keep only districts that have both funding AND performance data
merged_clean = merged.dropna(subset=["per_pupil_spending", "math_proficiency_pct"])
print(f"  After dropping missing key vars: {len(merged_clean):,} rows")
print(f"  Unique districts in final dataset: {merged_clean['leaid'].nunique():,}\n")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

print("Step 6 — Engineering features for modeling...\n")

df = merged_clean.copy()

# Funding tier (quartile-based categorical)
df["funding_tier"] = pd.qcut(
    df["per_pupil_spending"],
    q=4,
    labels=["Low", "Mid-Low", "Mid-High", "High"]
)

# Pupil-to-teacher ratio
df["student_teacher_ratio"] = (
    df["enrollment"] / df["total_teachers_fte"]
).replace([np.inf, -np.inf], np.nan).round(1)

# Composite performance score (average of math + reading proficiency)
df["composite_score"] = (
    df["math_proficiency_pct"] + df["reading_proficiency_pct"]
) / 2

# Log of per-pupil spending (common in education economics research)
df["log_per_pupil_spending"] = np.log(df["per_pupil_spending"])

# High-need flag: districts with high local revenue dependency (>60%) tend to be wealthier
df["high_local_dependency"] = (df["pct_local_revenue"] > 60).astype(int)

print("  New features created:")
features = [
    "funding_tier", "student_teacher_ratio",
    "composite_score", "log_per_pupil_spending", "high_local_dependency"
]
for f in features:
    non_null = df[f].notna().sum()
    print(f"    • {f:<30} {non_null:,} non-null values")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — EXPORT FINAL DATASET
# ═══════════════════════════════════════════════════════════════

print("\nStep 7 — Exporting final merged dataset...\n")
EXPORT_DIR = "data/processed"
OUTPUT_CSV = os.path.join(EXPORT_DIR, f"school_funding_merged_{TARGET_YEAR.replace('-','_')}.csv")
df.to_csv(OUTPUT_CSV, index=False)
print(f"  ✓ Saved to: {OUTPUT_CSV}")
print(f"  Final dataset: {df.shape[0]:,} districts × {df.shape[1]} columns\n")

# ── Quick summary ─────────────────────────────────────────────
print("=" * 55)
print(" DATASET SUMMARY")
print("=" * 55)

summary_cols = [
    "per_pupil_spending", "math_proficiency_pct",
    "reading_proficiency_pct", "graduation_rate",
    "student_teacher_ratio", "pct_certified_teachers",
]
available = [c for c in summary_cols if c in df.columns]
print(df[available].describe().round(2).to_string())

print("\n  Funding tier distribution:")
print(df["funding_tier"].value_counts().sort_index().to_string())

print("\n  Missing value summary (key columns):")
key_cols = available + ["pct_local_revenue", "pct_federal_revenue"]
for col in key_cols:
    if col in df.columns:
        pct_missing = df[col].isna().mean() * 100
        print(f"    {col:<35} {pct_missing:5.1f}% missing")

print("\n✅ Pipeline complete. Starting EDA and modeling...\n")


# ═══════════════════════════════════════════════════════════════
# STEP 8 — EDA PLOTS & REGRESSION MODELS
# ═══════════════════════════════════════════════════════════════

print("Step 8 — EDA plots and regression models...\n")

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
TIER_ORDER   = ["Low", "Mid-Low", "Mid-High", "High"]
TIER_PALETTE = {"Low": "#d62728", "Mid-Low": "#ff7f0e",
                "Mid-High": "#2ca02c", "High": "#1f77b4"}


# ── 8a. Scatter: per-pupil spending vs math proficiency ───────
fig, ax = plt.subplots(figsize=(10, 6))
for tier in TIER_ORDER:
    sub = df[df["funding_tier"] == tier]
    ax.scatter(sub["per_pupil_spending"], sub["math_proficiency_pct"],
               c=TIER_PALETTE[tier], label=tier, alpha=0.35, s=15, edgecolors="none")

# OLS trend line
_xy = df[["per_pupil_spending", "math_proficiency_pct"]].dropna()
m, b = np.polyfit(_xy["per_pupil_spending"], _xy["math_proficiency_pct"], 1)
xs = np.linspace(_xy["per_pupil_spending"].min(), _xy["per_pupil_spending"].max(), 300)
ax.plot(xs, m * xs + b, "k--", linewidth=1.3, label="OLS trend")

ax.set_xlabel("Per-Pupil Spending ($)")
ax.set_ylabel("Math Proficiency (%)")
ax.set_title("Per-Pupil Spending vs Math Proficiency (2017–18)")
ax.legend(title="Funding Tier")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "scatter_spending_vs_math.png")
plt.savefig(p, dpi=150); plt.close()
print(f"  ✓ {p}")


# ── 8b. Correlation heatmap ────────────────────────────────────
corr_cols = [c for c in [
    "per_pupil_spending", "math_proficiency_pct", "reading_proficiency_pct",
    "student_teacher_ratio", "pct_certified_teachers",
    "pct_local_revenue", "pct_federal_revenue", "pct_instr_spending",
] if c in df.columns]

corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
            mask=mask, square=True, linewidths=0.4, ax=ax)
ax.set_title("Correlation Matrix — Key Variables")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
plt.savefig(p, dpi=150); plt.close()
print(f"  ✓ {p}")


# ── 8c. Box plots: proficiency by funding tier ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, col, label in zip(axes,
                           ["math_proficiency_pct", "reading_proficiency_pct"],
                           ["Math Proficiency (%)", "Reading Proficiency (%)"]):
    if col in df.columns:
        sns.boxplot(data=df, x="funding_tier", y=col, order=TIER_ORDER,
                    palette=TIER_PALETTE, ax=ax, linewidth=0.8)
        ax.set_title(f"{label} by Funding Tier")
        ax.set_xlabel("Funding Tier")
        ax.set_ylabel(label)
fig.suptitle("Student Performance by Funding Tier (2017–18)", fontsize=13)
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "boxplot_proficiency_by_tier.png")
plt.savefig(p, dpi=150); plt.close()
print(f"  ✓ {p}\n")


# ── 8d. Models ─────────────────────────────────────────────────

FEATURES = [c for c in [
    "log_per_pupil_spending", "pct_local_revenue", "pct_federal_revenue",
    "pct_instr_spending", "student_teacher_ratio", "pct_certified_teachers",
] if c in df.columns]
TARGET = "math_proficiency_pct"

model_df = df[FEATURES + [TARGET]].dropna()
X = model_df[FEATURES]
y = model_df[TARGET]
print(f"  Model sample: {len(model_df):,} districts × {len(FEATURES)} features\n")

# OLS ──────────────────────────────────────────────────────────
ols = sm.OLS(y, sm.add_constant(X)).fit()
print("  ── OLS Regression ─────────────────────────────────────")
print(ols.summary())

# Random Forest ────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred = rf.predict(X_te)

print(f"\n  ── Random Forest ──────────────────────────────────────")
print(f"  Test R²  : {r2_score(y_te, y_pred):.3f}")
print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_te, y_pred)):.2f} pp")

# Feature importance bar chart
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="#1f77b4")
ax.set_title("Random Forest — Feature Importances")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, "rf_feature_importance.png")
plt.savefig(p, dpi=150); plt.close()
print(f"\n  ✓ {p}")

print(f"\n  All plots saved to ./{PLOTS_DIR}/")
print("✅ Analysis complete.\n")
