#!/usr/bin/env python3
"""Exploratory affinity/contact correlations for crystal_affinity_interactions CSV.
Only the Python standard library is required. No docking or RMSD involved.
"""
import argparse
import csv
import math
from statistics import mean, median

CONTACTS = (
    "hydrophobic", "hydrogen_bonds", "halogen_bonds", "ionic",
    "cation_pi", "pi_stacking", "metal", "total_interactions",
)
FEATURES = CONTACTS + ("satisfied_fraction",)


def safe_float(s):
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def ranks(xs):
    indexes = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    start = 0
    while start < len(xs):
        end = start + 1
        while end < len(xs) and xs[indexes[end]] == xs[indexes[start]]:
            end += 1
        value = (start + 1 + end) / 2.0
        for pos in range(start, end):
            out[indexes[pos]] = value
        start = end
    return out


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def spearman(xs, ys):
    return pearson(ranks(xs), ranks(ys))


def transformed_value(row, feature):
    value = safe_float(row.get(feature))
    if value is None:
        return None
    if feature == "satisfied_fraction":
        return value
    if feature.endswith("_per_heavy_atom"):
        original = feature.removesuffix("_per_heavy_atom")
        base = safe_float(row.get(original))
        atoms = safe_float(row.get("ligand_heavy_atoms"))
        return base / atoms if base is not None and atoms and atoms > 0 else None
    return value


def get_measure(row, feature):
    if feature.endswith("_per_heavy_atom"):
        base = safe_float(row.get(feature.removesuffix("_per_heavy_atom")))
        atoms = safe_float(row.get("ligand_heavy_atoms"))
        return base / atoms if base is not None and atoms and atoms > 0 else None
    return safe_float(row.get(feature))


def analyze_group(label, rows, threshold, output):
    print(f"\n{label}: {len(rows)} complexes")
    if len(rows) < threshold:
        print(f"  Too few records for correlation (minimum {threshold})")
        return
    features = FEATURES + tuple(f"{x}_per_heavy_atom" for x in CONTACTS)
    results = []
    for feature in features:
        pairs = [(safe_float(r["p_affinity"]), get_measure(r, feature)) for r in rows]
        pairs = [(pk, x) for pk, x in pairs if pk is not None and x is not None]
        if len(pairs) < threshold:
            continue
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        if rho is None:
            continue
        results.append((feature, len(pairs), rho))
        output.append({"group": label, "feature": feature, "n": len(pairs), "spearman_rho": f"{rho:.6f}"})
    print("  Spearman rho with experimental pKd/pKi (not a causal estimate):")
    for feature, n, rho in sorted(results, key=lambda it: -abs(it[2])):
        print(f"    {feature:<39} {rho:+.3f}  n={n}")

    # Compare contact presence between weak and strong binding quartiles.
    ordered = sorted(rows, key=lambda r: float(r["p_affinity"]))
    quartile_n = len(ordered) // 4
    low = ordered[:quartile_n]
    high = ordered[-quartile_n:]
    print("  Fraction with >=1 interaction: low-pK quartile vs high-pK quartile:")
    for feature in CONTACTS[:-1]:
        low_rate = sum((safe_float(r.get(feature)) or 0) > 0 for r in low) / len(low)
        high_rate = sum((safe_float(r.get(feature)) or 0) > 0 for r in high) / len(high)
        print(f"    {feature:<22} {low_rate:5.1%} vs {high_rate:5.1%}")
    print(f"  pK quartiles: low <= {low[-1]['p_affinity']}, high >= {high[0]['p_affinity']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="CSV produced by crystal_affinity_interactions")
    parser.add_argument("--out", default="crystal_affinity_correlations.csv")
    parser.add_argument("--min-n", type=int, default=30)
    args = parser.parse_args()
    with open(args.csv, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"p_affinity", "status", "assay_type", "ligand_heavy_atoms"} | set(FEATURES)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit("Missing input columns: " + ", ".join(sorted(missing)))
        all_rows = list(reader)
    rows = [r for r in all_rows if r["status"] == "ok" and safe_float(r["p_affinity"]) is not None]
    print(f"Read {len(all_rows)} records; valid {len(rows)}; errors {len(all_rows) - len(rows)}")
    results = []
    analyze_group("Kd and Ki together", rows, args.min_n, results)
    for kind in ("Kd", "Ki"):
        analyze_group(kind, [r for r in rows if r["assay_type"] == kind], args.min_n, results)
    with open(args.out, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "feature", "n", "spearman_rho"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCorrelation table saved to: {args.out}")
    print("Note: contact counts can correlate with ligand size; check normalized rows.")
    print("These are between-complex correlations, not evidence of causation or optimal score weights.")


if __name__ == "__main__":
    main()
