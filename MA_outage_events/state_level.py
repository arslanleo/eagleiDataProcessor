import pandas as pd
import os


DATA_FOLDER   = "Eaglei_data"         # where the raw CSVs live
OUTPUT_FOLDER = "cleaned_data_state"  # new output directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

STATE_NAME = "Massachusetts"             # ← pick your state here
YEAR_START, YEAR_END = 2018, 2023     # inclusive


all_years = []

for year in range(YEAR_START, YEAR_END + 1):
    fname = f"eaglei_outages_{year}.csv"
    fpath = os.path.join(DATA_FOLDER, fname)
    print(f"Processing {fname} …")

    try:
        df = pd.read_csv(fpath)

        # Harmonise the outage-count column
        if "customers_out" in df.columns:
            df["customers_out"] = df["customers_out"]
        elif "sum" in df.columns:       # 2023 files have sum instead of customers_out
            df["customers_out"] = df["sum"]
        else:
            print("  ⤬ No 'customers_out' or 'sum' column — skipped.")
            continue

        # ▸ Keep only the selected state
        df = df[df["state"] == STATE_NAME]

        if df.empty:
            print(f"  ⤬ No rows for {STATE_NAME} — skipped.")
            continue

        # Save the per-year state-level extract
        out_single = os.path.join(
            OUTPUT_FOLDER, f"Raw_state_{year}_{STATE_NAME}.xlsx"
        )
        df.to_excel(out_single, index=False)
        print(f"  ✓ Saved → {out_single}")

        all_years.append(df)

    except Exception as exc:
        print(f"  ⚠ Error reading {fname}: {exc}")
#Merge all the data together
if all_years:
    merged = pd.concat(all_years, ignore_index=True)
    merged.sort_values("run_start_time", inplace=True)
    out_merged = os.path.join(
        OUTPUT_FOLDER, f"Merged_Raw_Cleaned_state_{YEAR_START}_{YEAR_END}_{STATE_NAME}.xlsx"
    )
    merged.to_excel(out_merged, index=False)
    print(f"\nAll years merged → {out_merged}")
else:
    print("\nNo data processed; nothing to merge.")
