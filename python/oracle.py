# oracle.py — pure-Python float64 oracle for exocortex-embed-mojo
#
# Mirrors src/exocortex_embed/*.mojo op for op, including the EXACT
# floating-point evaluation order (SIMD lane blocking in blocks of 4,
# left-to-right accumulation, LCG draw order, Newton sqrt iterations,
# truncation-toward-zero quantization). With that, Python floats (IEEE-754
# binary64) are bit-for-bit comparable with Mojo Float64 results.
#
# Parity contract:
#   mojo  : mojo build -I src -D ASSERT=all examples/demo.mojo -o build/demo && build/demo
#   python: python3 python/oracle.py
#   both print a "PARITY" section of lines:  <case_name> <value>
#     - floats are printed as the decimal u64 of their IEEE754 bit pattern
#     - ints (quantized i8 / counts) are printed as plain decimals
#   python3 python/parity_check.py diffs the two blocks; empty diff = PASS.

import math
import struct

SIMD_WIDTH = 4
U64_MASK = (1 << 64) - 1


# --------------------------------------------------------------------------
# LCG (Numerical Recipes constants) — mirrors random_proj.mojo
# --------------------------------------------------------------------------
class LCG:
    def __init__(self, seed: int):
        self.state = seed & U64_MASK

    def next(self) -> float:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & U64_MASK
        raw = float(self.state >> 33) / float(1 << 31)
        return raw - 0.5


# --------------------------------------------------------------------------
# Newton sqrt — mirrors the module-local sqrt in random_proj.mojo
# --------------------------------------------------------------------------
def newton_sqrt(x: float) -> float:
    if x <= 0.0:
        return 0.0
    s = x
    for _ in range(20):
        s = 0.5 * (s + x / s)
    return s


# --------------------------------------------------------------------------
# vector ops — mirrors vector.mojo (incl. lane-blocked summation order)
# --------------------------------------------------------------------------
def dot(a, b):
    n = len(a)
    total = 0.0
    i = 0
    while i + SIMD_WIDTH <= n:
        lane_sum = 0.0
        for j in range(SIMD_WIDTH):
            lane_sum = lane_sum + a[i + j] * b[i + j]
        total = total + lane_sum
        i += SIMD_WIDTH
    while i < n:
        total = total + a[i] * b[i]
        i += 1
    return total


def norm(v):
    return math.sqrt(dot(v, v))


def cosine_similarity(a, b):
    denom = norm(a) * norm(b)
    if denom == 0.0:
        return 0.0
    return dot(a, b) / denom


def euclidean_distance(a, b):
    n = len(a)
    sum_sq = 0.0
    i = 0
    while i + SIMD_WIDTH <= n:
        lane_sum = 0.0
        for j in range(SIMD_WIDTH):
            diff = a[i + j] - b[i + j]
            lane_sum = lane_sum + diff * diff
        sum_sq = sum_sq + lane_sum
        i += SIMD_WIDTH
    while i < n:
        diff = a[i] - b[i]
        sum_sq = sum_sq + diff * diff
        i += 1
    return math.sqrt(sum_sq)


def add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def scale(v, s):
    return [x * s for x in v]


def normalize(v):
    n = norm(v)
    if n == 0.0:
        return [0.0] * len(v)
    return scale(v, 1.0 / n)


# --------------------------------------------------------------------------
# matrix ops — mirrors matrix.mojo (pre-transpose B, lane-ordered dot)
# --------------------------------------------------------------------------
def matmul(A, B):
    m = len(A)
    p = len(A[0])
    n = len(B[0])
    BT = [[B[i][j] for i in range(p)] for j in range(n)]
    result = []
    for i in range(m):
        row = []
        for j in range(n):
            row.append(dot(A[i], BT[j]))
        result.append(row)
    return result


def transpose(A):
    m = len(A)
    if m == 0:
        return []
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


# --------------------------------------------------------------------------
# random projection — mirrors random_proj.mojo
# --------------------------------------------------------------------------
def init_random_matrix(input_dim, output_dim, seed):
    rng = LCG(seed)
    scale_f = 1.0 / newton_sqrt(float(output_dim))
    matrix = []
    for _ in range(output_dim):
        row = []
        for _ in range(input_dim):
            s = 0.0
            for _ in range(12):
                s = s + (rng.next() + 0.5)
            row.append((s - 6.0) * scale_f)
        matrix.append(row)
    return matrix


def project(input_dim, output_dim, seed, v):
    P = init_random_matrix(input_dim, output_dim, seed)
    # PT[j][i] = P[i][j]  (input_dim x output_dim)
    PT = [[P[i][j] for i in range(output_dim)] for j in range(input_dim)]
    return matmul([list(v)], PT)[0]


# --------------------------------------------------------------------------
# scalar quantizer — mirrors quantize.mojo
# --------------------------------------------------------------------------
class ScalarQuantizer:
    def __init__(self, dim: int):
        self.dim = dim
        self.mins = [0.0] * dim
        self.maxs = [0.0] * dim
        self.fitted = False

    def fit(self, vectors):
        if len(vectors) == 0:
            return
        for j in range(self.dim):
            self.mins[j] = 1e308
            self.maxs[j] = -1e308
        for vec in vectors:
            for j in range(self.dim):
                val = vec[j]
                if val < self.mins[j]:
                    self.mins[j] = val
                if val > self.maxs[j]:
                    self.maxs[j] = val
        self.fitted = True

    def quantize(self, v):
        out = []
        for j in range(self.dim):
            span = self.maxs[j] - self.mins[j]
            if span == 0.0:
                out.append(0)
            else:
                normalized = (v[j] - self.mins[j]) / span * 255.0
                q = int(normalized) - 128  # int() truncates toward zero, as Mojo Int(f64)
                q = max(-128, min(127, q))
                out.append(q)
        return out

    def dequantize(self, q):
        out = []
        for j in range(self.dim):
            span = self.maxs[j] - self.mins[j]
            out.append(self.mins[j] + (float(q[j]) + 128.0) * span / 255.0)
        return out

    def quantized_distance(self, q1, q2):
        sum_sq = 0.0
        for j in range(self.dim):
            diff = float(q1[j] - q2[j])
            sum_sq = sum_sq + diff * diff
        return sum_sq


# --------------------------------------------------------------------------
# vector index — mirrors index.mojo (insertion-order scan, first-max selection)
# --------------------------------------------------------------------------
class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.entries = []  # list of (id, vector)

    def add(self, id, vector):
        self.entries.append((id, list(vector)))

    def search(self, query, k):
        scores = [(eid, cosine_similarity(query, vec)) for (eid, vec) in self.entries]
        used = [False] * len(scores)
        result = []
        for _ in range(min(k, len(scores))):
            best_idx = -1
            best_score = -2.0
            for i in range(len(scores)):
                if (not used[i]) and scores[i][1] > best_score:
                    best_score = scores[i][1]
                    best_idx = i
            if best_idx >= 0:
                used[best_idx] = True
                result.append(scores[best_idx])
        return result

    def size(self):
        return len(self.entries)


# --------------------------------------------------------------------------
# shared scenario — the demo's cases (used by both receipt and PARITY block)
# --------------------------------------------------------------------------
def f64_bits(x: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", x))[0]


def scenario():
    """Runs every demo case; returns list of (name, pretty, parity_value).

    parity_value: float -> printed as u64 bit-decimal; int -> printed as int.
    """
    lines = []

    def F(name, val):
        lines.append((name, repr(val), str(f64_bits(val))))

    def I(name, val):
        lines.append((name, str(val), str(val)))

    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0]

    # --- vector ops ---
    F("dot4", dot(a, b))
    v3a = [1.0, 2.0, 3.0]
    v3b = [4.0, 5.0, 6.0]
    F("dot3_tail", dot(v3a, v3b))                       # 32.0
    a7 = [float(i) for i in range(7)]
    b7 = [float(i + 1) for i in range(7)]
    F("dot7_lane_tail", dot(a7, b7))
    F("dot_empty", dot([], []))                          # 0.0 degenerate
    F("norm_340", norm([3.0, 4.0, 0.0]))                 # 5.0
    F("cos_orth", cosine_similarity([1.0, 0.0], [0.0, 1.0]))
    F("cos_ident", cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    F("cos_opp", cosine_similarity([1.0, 0.0], [-1.0, 0.0]))
    F("cos_zero", cosine_similarity([0.0, 0.0], [1.0, 0.0]))  # degenerate
    F("euc_34", euclidean_distance([0.0, 0.0], [3.0, 4.0]))
    summed = add(a, b)
    for i, x in enumerate(summed):
        F(f"add4_e{i}", x)
    scaled = scale([1.0, 2.0, 3.0], 2.0)
    for i, x in enumerate(scaled):
        F(f"scale3_e{i}", x)
    nrm = normalize([3.0, 4.0])
    for i, x in enumerate(nrm):
        F(f"norm34_e{i}", x)                             # note: e0 = 0.6000000000000001
    nrm0 = normalize([0.0, 0.0])
    for i, x in enumerate(nrm0):
        F(f"normzero_e{i}", x)                           # degenerate

    # --- matrix ops ---
    A2 = [[1.0, 2.0], [3.0, 4.0]]
    B2 = [[5.0, 6.0], [7.0, 8.0]]
    C = matmul(A2, B2)
    for i in range(2):
        for j in range(2):
            F(f"mm2_{i}{j}", C[i][j])
    I3 = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    A3 = [[float(i * 3 + j + 1) for j in range(3)] for i in range(3)]
    R = matmul(A3, I3)
    F("mmI_00", R[0][0])
    F("mmI_12", R[1][2])
    F("mmI_22", R[2][2])

    # --- random projection ---
    high = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    low = project(8, 3, 12345, high)
    for i, x in enumerate(low):
        F(f"rp8_e{i}", x)
    ones5 = [1.0] * 5
    p5 = project(5, 2, 999, ones5)
    for i, x in enumerate(p5):
        F(f"rp5_e{i}", x)
    p0 = project(4, 2, 7, [0.0, 0.0, 0.0, 0.0])          # degenerate zero input
    for i, x in enumerate(p0):
        F(f"rpzero_e{i}", x)

    # --- scalar quantization ---
    v1 = [0.1, 0.2, 0.3, 0.4]
    v2 = [0.9, 0.8, 0.7, 0.6]
    v3 = [0.5, 0.5, 0.5, 0.5]
    q = ScalarQuantizer(4)
    q.fit([v1, v2, v3])
    qv1 = q.quantize(v1)
    for i, x in enumerate(qv1):
        I(f"q_v1_e{i}", x)
    dq1 = q.dequantize(qv1)
    for i, x in enumerate(dq1):
        F(f"dq_v1_e{i}", x)
    qs = ScalarQuantizer(2)
    qs.fit([[0.0, 10.0]])
    qsv = qs.quantize([0.0, 10.0])
    for i, x in enumerate(qsv):
        I(f"q_single_e{i}", x)                           # zero span -> 0
    dqs = qs.dequantize(qsv)
    for i, x in enumerate(dqs):
        F(f"dq_single_e{i}", x)

    # --- vector index ---
    idx = VectorIndex(4)
    idx.add("doc1", v1)
    idx.add("doc2", v2)
    idx.add("doc3", v3)
    query = [0.15, 0.25, 0.35, 0.45]
    results = idx.search(query, 3)
    for name, score in results:
        F(f"idx_score_{name}", score)
    empty_idx = VectorIndex(3)
    I("idx_empty_size", empty_idx.size())                # degenerate
    I("idx_empty_results", len(empty_idx.search([1.0, 0.0, 0.0], 5)))

    return lines


def main():
    print("=== exocortex-embed-mojo oracle receipt (python) ===")
    print()
    print("--- PARITY ---")
    for name, _pretty, parity in scenario():
        print(name, parity)
    print("--- END PARITY ---")


if __name__ == "__main__":
    main()
