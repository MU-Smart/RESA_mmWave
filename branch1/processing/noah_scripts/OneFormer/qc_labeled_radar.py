import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labeled_csv)
    print("Rows:", len(df))
    print("\nBucket distribution:")
    print(df["bucket"].value_counts(dropna=False))
    print("\nADE distribution (top 20):")
    print(df["ade_name"].value_counts(dropna=False).head(20))

    # Useful diagnostics
    if "maj_frac" in df.columns:
        print("\nmaj_frac stats:", df["maj_frac"].describe())
    if "cam_z" in df.columns:
        print("\ncam_z stats:", df["cam_z"].describe())

if __name__ == "__main__":
    main()