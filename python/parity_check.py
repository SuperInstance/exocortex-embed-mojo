# parity_check.py — bit-for-bit parity contract for exocortex-embed-mojo
#
# Runs the Mojo demo receipt and the pure-Python oracle, extracts the PARITY
# block from each (lines: `name <u64-bit-decimal | int>`), and diffs them.
# Empty diff = bit-for-bit identical Float64 results.
#
# Usage:
#   python3 python/parity_check.py [--mojo build/demo]
# Exits 0 on parity, 1 on any mismatch.
# (Fleet precedent: grand-pattern-mojo `make parity` contract.)

import argparse
import subprocess
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_block(text):
    """Returns list of PARITY lines between --- PARITY --- and --- END PARITY ---."""
    lines = []
    inside = False
    for line in text.splitlines():
        if line.strip() == "--- PARITY ---":
            inside = True
            continue
        if line.strip() == "--- END PARITY ---":
            inside = False
            continue
        if inside and line.strip():
            lines.append(line.strip())
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mojo", default=os.path.join(REPO, "build", "demo"),
                        help="path to built examples/demo.mojo binary")
    args = parser.parse_args()

    if not os.path.exists(args.mojo):
        print(f"ERROR: {args.mojo} not found. Build it first:")
        print("  mojo build -I src -D ASSERT=all examples/demo.mojo -o build/demo")
        return 1

    mojo_out = subprocess.run(
        [args.mojo], capture_output=True, text=True, check=True
    ).stdout

    oracle_path = os.path.join(REPO, "python", "oracle.py")
    py_out = subprocess.run(
        [sys.executable, oracle_path], capture_output=True, text=True, check=True
    ).stdout

    mojo_lines = extract_block(mojo_out)
    py_lines = extract_block(py_out)

    mojo_map = dict(l.split(None, 1) for l in mojo_lines)
    py_map = dict(l.split(None, 1) for l in py_lines)

    names_m = [l.split(None, 1)[0] for l in mojo_lines]
    names_p = [l.split(None, 1)[0] for l in py_lines]

    print(f"cases: mojo={len(mojo_lines)}  python={len(py_lines)}")

    missing = sorted(set(names_p) - set(names_m)) + sorted(set(names_m) - set(names_p))
    if missing:
        print("MISSING-CASE:", ", ".join(missing))

    order_issue = names_m != names_p
    if order_issue:
        print("NOTE: case order differs between blocks (comparing by name)")

    mismatches = []
    print()
    print(f"{'case':<22} {'mojo':>22} {'python':>22}  match")
    for name in names_p:
        if name not in mojo_map:
            continue
        m, p = mojo_map[name], py_map[name]
        ok = m == p
        if not ok:
            mismatches.append(name)
        print(f"{name:<22} {m:>22} {p:>22}  {'OK' if ok else 'MISMATCH'}")

    print()
    total = len(names_p)
    good = total - len([n for n in mismatches if n in set(names_p)])
    if mismatches or missing or order_issue:
        print(f"PARITY: FAIL — {good}/{total} bit-identical, "
              f"{len(mismatches)} mismatches, missing={len(missing)}, order_issue={order_issue}")
        return 1
    print(f"PARITY: PASS — {total}/{total} cases bit-for-bit identical (IEEE754 f64)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
