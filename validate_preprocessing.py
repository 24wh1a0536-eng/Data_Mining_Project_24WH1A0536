"""
validate_preprocessing.py
Beginner-friendly validation for the Pima Diabetes ARFF dataset.

What it does:
1. Reads an ARFF file without requiring extra Python packages.
2. Reports attributes, number of rows, missing values, class distribution,
   and suspicious zero values in selected medical measurements.
3. Creates a standardized CSV copy of numeric input attributes for inspection.
4. Checks that row count and class labels are preserved after preprocessing.

Important:
- This script is for checking/learning, not medical diagnosis.
- It does NOT overwrite the original ARFF file.
- Use the original ARFF in Weka unless you intentionally want to import the
  generated CSV. Do not apply Normalize + Standardize again without a reason.
"""

from pathlib import Path
import csv
import math
import statistics
import sys

# Change this if your file is stored somewhere else.
INPUT_FILE = Path("diabetes.arff")
OUTPUT_FILE = Path("diabetes_preprocessed.csv")

# In the Pima dataset, zero can be suspicious for these measurements.
# Zero pregnancies can be valid, so it is intentionally not listed here.
ZERO_CHECK_ATTRIBUTES = {"plas", "pres", "skin", "insu", "mass"}


def parse_arff(path):
    attributes = []
    rows = []
    in_data = False

    with path.open("r", encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("%"):
                continue

            lower = line.lower()
            if not in_data:
                if lower.startswith("@attribute"):
                    # Attribute name may be quoted or unquoted.
                    rest = line[len("@attribute"):].strip()
                    if rest.startswith("'") or rest.startswith('"'):
                        quote = rest[0]
                        end = rest.find(quote, 1)
                        if end == -1:
                            raise ValueError(f"Could not parse attribute: {line}")
                        name = rest[1:end]
                        kind = rest[end + 1:].strip()
                    else:
                        pieces = rest.split(None, 1)
                        if len(pieces) != 2:
                            raise ValueError(f"Could not parse attribute: {line}")
                        name, kind = pieces
                    attributes.append((name, kind))
                elif lower == "@data":
                    in_data = True
            else:
                # This project dataset uses ordinary comma-separated ARFF rows.
                values = next(csv.reader([line], skipinitialspace=True))
                values = [v.strip().strip("'").strip('"') for v in values]
                if len(values) != len(attributes):
                    raise ValueError(
                        f"Row has {len(values)} values but expected "
                        f"{len(attributes)}: {line}"
                    )
                rows.append(values)

    if not attributes or not in_data:
        raise ValueError("No ARFF attributes or @data section found.")
    return attributes, rows


def is_numeric(kind):
    return kind.lower() in {"numeric", "real", "integer"}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_FILE
    output_path = path.with_name(OUTPUT_FILE.name)

    if not path.exists():
        print(f"ERROR: Could not find {path.resolve()}")
        print("Place diabetes.arff in this folder, or run:")
        print("  python validate_preprocessing.py \"C:\\path\\to\\diabetes.arff\"")
        raise SystemExit(1)

    attributes, rows = parse_arff(path)
    names = [name for name, _ in attributes]
    kinds = {name: kind for name, kind in attributes}
    if not rows:
        raise ValueError("The ARFF file contains no data rows.")

    print("=" * 62)
    print("DIABETES DATASET: PREPROCESSING VALIDATION")
    print("=" * 62)
    print(f"Input file: {path.resolve()}")
    print(f"Rows (instances): {len(rows)}")
    print(f"Attributes: {len(attributes)}")
    print("\nAttributes:")
    for name, kind in attributes:
        print(f"  - {name}: {kind}")

    missing_by_attribute = {}
    for index, name in enumerate(names):
        missing_by_attribute[name] = sum(
            1 for row in rows if row[index] == "?" or row[index] == ""
        )

    print("\nMissing values by attribute:")
    any_missing = False
    for name, count in missing_by_attribute.items():
        print(f"  {name}: {count}")
        any_missing = any_missing or count > 0
    if not any_missing:
        print("  No missing values marked as '?' or blank.")

    class_name = names[-1]
    class_index = len(names) - 1
    class_counts = {}
    for row in rows:
        label = row[class_index]
        class_counts[label] = class_counts.get(label, 0) + 1

    print(f"\nClass attribute (assumed last): {class_name}")
    print("Class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count} ({count / len(rows) * 100:.2f}%)")

    print("\nSuspicious zero checks:")
    found_zero = False
    for name in ZERO_CHECK_ATTRIBUTES.intersection(names):
        index = names.index(name)
        count = sum(
            1 for row in rows
            if row[index] not in {"?", ""}
            and is_numeric(kinds[name])
            and float(row[index]) == 0.0
        )
        print(f"  {name}: {count} zero value(s)")
        found_zero = found_zero or count > 0
    if not found_zero:
        print("  No zero values found in the selected measurement attributes.")

    # Standardize numeric input columns only; leave the class label unchanged.
    numeric_input_indices = [
        i for i, (name, kind) in enumerate(attributes[:-1])
        if is_numeric(kind)
    ]
    output_rows = [row[:] for row in rows]
    scaling_stats = {}

    for i in numeric_input_indices:
        name = names[i]
        observed = []
        for row in rows:
            if row[i] not in {"?", ""}:
                observed.append(float(row[i]))

        mean = statistics.mean(observed) if observed else 0.0
        std = statistics.pstdev(observed) if len(observed) > 1 else 0.0
        scaling_stats[name] = (mean, std)

        for row_index, row in enumerate(rows):
            value = row[i]
            if value in {"?", ""}:
                output_rows[row_index][i] = ""
            elif std == 0:
                output_rows[row_index][i] = "0"
            else:
                output_rows[row_index][i] = f"{(float(value) - mean) / std:.8f}"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(output_rows)

    # Validate that the preprocessing preserved rows and target labels.
    output_class_counts = {}
    for row in output_rows:
        label = row[class_index]
        output_class_counts[label] = output_class_counts.get(label, 0) + 1

    print("\nPreprocessing validation:")
    print(f"  Original row count: {len(rows)}")
    print(f"  Processed row count: {len(output_rows)}")
    print(f"  Row count preserved: {'PASS' if len(rows) == len(output_rows) else 'FAIL'}")
    print(f"  Class labels/distribution preserved: "
          f"{'PASS' if class_counts == output_class_counts else 'FAIL'}")
    print(f"  Numeric input attributes standardized: {len(numeric_input_indices)}")
    print(f"  Class attribute transformed: NO")
    print(f"\nCreated processed CSV: {output_path.resolve()}")
    print("\nNotes:")
    print("- The original ARFF file was not changed.")
    print("- Blank values in the CSV represent missing values; this script does not impute them.")
    print("- Zero checks flag values for review; they are not automatically changed.")
    print("- For fair model evaluation, fit scaling inside each cross-validation fold.")
    print("- If using Weka, keep the same preprocessing and 10-fold CV settings for all classifiers.")


if __name__ == "__main__":
    main()
