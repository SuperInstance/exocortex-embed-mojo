# USERMANUAL — exocortex-embed-mojo on this box

**Status: VERIFIED WORKING (2026-10-01, Mojo 1.2.0.dev2026100105).**
SIMD-styled embedding math for the exocortex: dot/norm/cosine/euclidean,
matmul/transpose, JL random projection (LCG + CLT), brute-force cosine index,
f64→i8 scalar quantization. 68 test assertions, 52-case bit-for-bit Python
parity, 18-check demo receipt.

---

## Quickstart

This repo has **no pixi manifest**. The proven toolchain on this box is
quilt-mojo-lab's pixi environment, driven via env vars:

```bash
export MODULAR_HOME=$HOME/projects/quilt-mojo-lab/.pixi/envs/default/share/max
export PATH=$HOME/projects/quilt-mojo-lab/.pixi/envs/default/bin:$PATH

cd exocortex-embed-mojo

make test      # all 5 test suites, -D ASSERT=all, flags BEFORE filename
make demo      # examples/demo.mojo receipt (inputs, outputs, PASS/FAIL)
make parity    # demo vs python/oracle.py, bit-for-bit IEEE754 diff
```

Direct invocation is identical, just with the env vars:

```bash
mojo build -I src -D ASSERT=all tests/test_vector.mojo -o build/test_vector && build/test_vector
```

`-I src` (not `-I .`) — the package lives at `src/exocortex_embed/` with an
`__init__.mojo`. In the Makefile, `export MODULAR_HOME` must actually say
`export`; a bare `MODULAR_HOME :=` assignment silently produces
`unable to locate module 'std'`.

---

## What it is

| Module | Contents |
|--------|----------|
| `src/exocortex_embed/vector.mojo` | dot, norm, cosine, euclidean, add, scale, normalize — lane-blocked (width 4) accumulation |
| `src/exocortex_embed/matrix.mojo` | matmul (pre-transposes B), transpose, row, col |
| `src/exocortex_embed/random_proj.mojo` | JL projection: LCG (Numerical Recipes constants) + 12-uniform CLT Gaussian, 1/√k scale, module-local Newton sqrt |
| `src/exocortex_embed/index.mojo` | VectorIndex: add/remove/search top-k cosine, first-max selection sort |
| `src/exocortex_embed/quantize.mojo` | ScalarQuantizer: per-dim min/max fit, f64→i8 (truncate, not round), dequantize, int8-space distance |
| `src/main.mojo` | original demo (same content as examples/demo.mojo, minus receipt/parity) |
| `examples/demo.mojo` | receipt + PASS/FAIL checks + PARITY block driver |
| `tests/test_*.mojo` | 5 suites, self-reporting `Counters` |
| `python/oracle.py` | pure-Python f64 oracle, mirrors Mojo op-for-op |
| `python/parity_check.py` | diffs the PARITY blocks |

---

## Test suite

- `test_vector.mojo` — **15 assertions** (14 pass, 1 booked-fail, see below)
- `test_matrix.mojo` — **15 assertions**
- `test_random_proj.mojo` — **9 assertions**
- `test_index.mojo` — **13 assertions**
- `test_quantize.mojo` — **16 assertions**
- **Total: 68 assertions, 67 pass, 1 booked-fail** (README claimed 52 — the
  original counts were aspirational and never executed; actual counts differ)

Expected output per suite ends:

```
Results:  N  passed,  M  failed
ALL TESTS PASSED        (or TESTS FAILED)
```

Suites are self-reporting (`Counters.record`) because the builtin
`assert(cond, msg)` **cannot be called** in Mojo 1.2 (see traps). Run with
and without `-D ASSERT=all` — results are identical; the flag is kept per
fleet convention.

Verified this box: 67/68 PASS with and without the flag; the single FAIL is
booked, not a porting artifact (below).

---

## The one booked test failure (NOT fixed, fleet rule)

`test_vector.mojo::test_normalize` asserts `normalize([3,4])[0] == 0.6`
exactly. Reality: `normalize` computes `3.0 * (1.0 / 5.0)` = **0.6000000000000001**
in IEEE-754 f64 — on every platform, in any language, since forever. The
original constants were written without ever running the code. The Python
oracle produces the identical bits (parity case `norm34_e0`), so this is an
aspirational test constant, **not** a porting or math bug. Booked; suite
honestly reports 14/15.

---

## Parity contract

`python/oracle.py` is a pure-Python float64 oracle mirroring the Mojo ops
**including evaluation order**: lane-blocked dot accumulation (blocks of 4,
left-to-right within lane), LCG 64-bit wrap arithmetic, 12-uniform CLT draw
order, module-local Newton sqrt (20 iterations), truncation-toward-zero
quantization, fit min/max scan order.

```bash
make parity    # → PARITY: PASS — 52/52 cases bit-for-bit identical (IEEE754 f64)
```

52 cases cover: dot (lane+tail+empty), norm, cosine (orthogonal / identical /
opposite / zero-vector), euclidean, add, scale, normalize (incl. zero vector),
matmul (known + identity), random projection (8→3, 5→2, zero input), quantize
/ dequantize (min-vector and zero-span single-vector), index scores and empty
index. Floats compare as the decimal u64 of their IEEE754 bit pattern
(`bitcast[DType.uint64]` on the Mojo side, `struct.pack('>d')` on the Python
side) — no float-formatting drift can masquerade as parity.

**This contract also verifies the hardcoded test constants against the live
oracle** — that's how the `0.6` booking above was proven to be an f64 reality
rather than a code defect.

---

## This-box receipt (2026-10-01)

```
=== exocortex-embed-mojo demo receipt ===
dot(a,b)=70  norm=5.477225575051661  cosine=0.9688639316269662  euclidean=8.0
matmul [[19,22],[43,50]]   transpose [[1,3],[2,4]]
index top-3: doc1 0.9979654098963515 / doc3 0.9370425713316363 / doc2 0.8753123968919349
8D→3D projection (seed 12345): [0.6838709561527941, 0.4152047028091609, 1.0667255538287395]
quantize(doc1) = [-128,-128,-128,-128] → dequantize [0.1,0.2,0.3,0.4]
=== summary: 18 passed, 0 failed ===
DEMO: ALL CHECKS PASSED
PARITY: PASS — 52/52 cases bit-for-bit identical (IEEE754 f64)
```

---

## Mojo 1.2 port notes (mechanical drift applied)

- `fn` → `def`, `let` → `var` (both removed from the language)
- `inout self` → `out self` in `__init__`, `mut self` in methods; read-only
  args drop `inout` entirely (default read convention)
- Module-level `alias` AND module-level `var` both banned
- `from math import sqrt` → `from std import math` + `math.sqrt`
- List returns / appends need explicit `^` transfers; `List.copy()` exists for
  explicit deep copy; `owned` parameter convention is GONE (parse error) —
  take read params and `.copy()` inside
- Tuple access is subscript (`t[0]`), not `.get[i]()`; Tuples are not
  implicitly copyable — rebuild from elements to store
- Cannot move out of a subscript through a `mut` ref (`self.entries[i]^`
  rejected) — rebuild entries via the copying constructor instead
- Calling a `mut` method from `__init__` requires all fields pre-initialized
  (assign a placeholder first)
- `bitcast[DType.uint64](x)` (takes a DType parameter, not a type)
- Method-call ban in demo: no variadic `List(1.0, 2.0)` init — use builders

## Booked, not fixed (semantic, originals in git history at 1907c36)

1. **test_index `test_remove` typo**: `v2.append(0.0); v1.append(1.0)` —
   clearly meant `v2.append(1.0)`. As written, entry "b" has 1 dim in a 2-dim
   index and the post-remove search would read out of bounds — but passes
   because cosine's zero-norm guard (`denom == 0 → return 0`) fires before
   dot() touches the short vector. Masked, alive, booked.
2. **normalize `0.6` test constant** (above).
3. **quantize truncates, README says round**: `Int(normalized)` truncates
   toward zero; README formulas say `round(...)`. Behavior is consistent
   Mojo↔Python (booked parity); docs drift only.
4. **README assertion counts (52)** don't match actuals (68); README SIMD
   claims are aspirational — the "SIMD" loops are manually unrolled scalar
   code, the compiler may or may not vectorize. No benchmarks were run (out
   of scope; CPU-only box, correctness mission).

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `unable to locate module 'exocortex_embed'` | Wrong include: use `-I src` (package is `src/exocortex_embed/`), not `-I .` |
| `unable to locate module 'std'` under `make` | Makefile forgot `export MODULAR_HOME` (bare make-var assignment isn't exported) |
| `pixi run mojo` fails here | No pixi manifest in this repo — use the MODULAR_HOME+PATH route (above) |
| `assert(cond, msg)` → "cannot be converted from 'Tuple[Bool, String]'" | Builtin assert is uncallable in 1.2; use the Counters pattern |
| `global variables are not supported` | Module-level `var`/`alias` banned; move state into a struct passed as `mut` |
| `'List[...]' cannot be implicitly copied` | Transfer with `^`, or `.copy()` explicitly; appends of locals need `append(x^)` |
| `cannot transfer out of immutable reference` | Moving out of read params/subscripts is banned — copy via constructor instead |
| `use of uninitialized value 'self...'` | `__init__` calling methods before all fields assigned — pre-initialize fields |
