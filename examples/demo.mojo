# demo.mojo — working example + parity driver for exocortex-embed-mojo
#
# Prints a human receipt (inputs → outputs → PASS/FAIL checks), then a
# machine PARITY block: one line per case, `name <u64>` where u64 is the
# decimal IEEE754 bit pattern of the Float64 result (ints printed as-is).
#
# python/oracle.py emits the identical block; python/parity_check.py diffs
# them. Empty diff = bit-for-bit parity between Mojo and the Python oracle.
#
# Mojo 1.2 port notes: module-level var counters banned -> Checks struct;
# bitcast is `bitcast[DType.uint64](x)`; no variadic List init -> builders.

from std.memory import bitcast
from exocortex_embed.vector import (
    dot, norm, cosine_similarity, euclidean_distance,
    add, scale, normalize,
)
from exocortex_embed.matrix import matmul, transpose
from exocortex_embed.random_proj import RandomProjection
from exocortex_embed.index import VectorIndex
from exocortex_embed.quantize import ScalarQuantizer


struct Checks(ImplicitlyCopyable):
    var passed: Int
    var failed: Int

    def __init__(out self):
        self.passed = 0
        self.failed = 0

    def check(mut self, cond: Bool, name: String):
        if cond:
            self.passed += 1
            print("  [PASS] ", name)
        else:
            self.failed += 1
            print("  [FAIL] ", name)


def bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def copy_vec(values: List[Float64]) -> List[Float64]:
    var result = List[Float64](capacity=len(values))
    for i in range(len(values)):
        result.append(values[i])
    return result^


def of2(a: Float64, b: Float64) -> List[Float64]:
    var v = List[Float64](capacity=2)
    v.append(a); v.append(b)
    return v^


def of3(a: Float64, b: Float64, c: Float64) -> List[Float64]:
    var v = List[Float64](capacity=3)
    v.append(a); v.append(b); v.append(c)
    return v^


def of5(a: Float64, b: Float64, c: Float64, d: Float64, e: Float64) -> List[Float64]:
    var v = List[Float64](capacity=5)
    v.append(a); v.append(b); v.append(c); v.append(d); v.append(e)
    return v^


def of4(a: Float64, b: Float64, c: Float64, d: Float64) -> List[Float64]:
    var v = List[Float64](capacity=4)
    v.append(a); v.append(b); v.append(c); v.append(d)
    return v^


def of7(a: Float64, b: Float64, c: Float64, d: Float64,
        e: Float64, f: Float64, g: Float64) -> List[Float64]:
    var v = List[Float64](capacity=7)
    v.append(a); v.append(b); v.append(c); v.append(d)
    v.append(e); v.append(f); v.append(g)
    return v^


def of8(a: Float64, b: Float64, c: Float64, d: Float64,
        e: Float64, f: Float64, g: Float64, h: Float64) -> List[Float64]:
    var v = List[Float64](capacity=8)
    v.append(a); v.append(b); v.append(c); v.append(d)
    v.append(e); v.append(f); v.append(g); v.append(h)
    return v^


def print_vec(label: String, v: List[Float64]):
    print(label, " = [", end="")
    for i in range(len(v)):
        if i > 0:
            print(", ", end="")
        print(v[i], end="")
    print("]")


def main() raises:
    var ck = Checks()
    print("=== exocortex-embed-mojo demo receipt ===")
    print()

    # -------------------------------------------------- vector ops
    print("--- Vector Operations ---")
    var a = of4(1.0, 2.0, 3.0, 4.0)
    var b = of4(5.0, 6.0, 7.0, 8.0)

    print("inputs: a = [1, 2, 3, 4]   b = [5, 6, 7, 8]")
    print("dot(a, b) = ", dot(a, b))
    print("norm(a)   = ", norm(a))
    print("cosine(a, b) = ", cosine_similarity(a, b))
    print("euclidean(a, b) = ", euclidean_distance(a, b))
    print_vec("add(a, b)", add(a, b))
    print_vec("normalize([3, 4])", normalize(of2(3.0, 4.0)))
    print_vec("normalize([0, 0])  (degenerate)", normalize(of2(0.0, 0.0)))

    ck.check(dot(a, b) == 70.0, "dot(a,b) == 70")
    ck.check(dot(of3(1.0, 2.0, 3.0), of3(4.0, 5.0, 6.0)) == 32.0,
             "dot tail case == 32")
    ck.check(norm(of3(3.0, 4.0, 0.0)) == 5.0, "norm([3,4,0]) == 5")
    ck.check(cosine_similarity(of2(1.0, 0.0), of2(0.0, 1.0)) == 0.0,
             "cosine orthogonal == 0")
    ck.check(cosine_similarity(of3(1.0, 2.0, 3.0), of3(1.0, 2.0, 3.0)) == 1.0,
             "cosine identical == 1")
    ck.check(cosine_similarity(of2(1.0, 0.0), of2(-1.0, 0.0)) == -1.0,
             "cosine opposite == -1")
    ck.check(cosine_similarity(of2(0.0, 0.0), of2(1.0, 0.0)) == 0.0,
             "cosine zero-vector guard == 0")
    ck.check(euclidean_distance(of2(0.0, 0.0), of2(3.0, 4.0)) == 5.0,
             "euclidean == 5")
    # NOTE: 3.0 * (1.0/5.0) = 0.6000000000000001 in f64 — IEEE reality, not a
    # bug. The Python oracle agrees bit-for-bit (see PARITY norm34_e0).
    ck.check(bits(normalize(of2(3.0, 4.0))[0]) == UInt64(4603579539098121012),
             "normalize[0] == f64(0.6000000000000001) exactly")
    print()

    # -------------------------------------------------- matrix ops
    print("--- Matrix Operations ---")
    var A = List[List[Float64]](capacity=2)
    A.append(of2(1.0, 2.0))
    A.append(of2(3.0, 4.0))
    var B = List[List[Float64]](capacity=2)
    B.append(of2(5.0, 6.0))
    B.append(of2(7.0, 8.0))
    print("inputs: A = [[1,2],[3,4]]   B = [[5,6],[7,8]]")
    var C = matmul(A, B)
    print("matmul(A, B) = [[", C[0][0], ", ", C[0][1], "], [", C[1][0], ", ", C[1][1], "]]")

    var AT = transpose(A)
    print("transpose(A) = [[", AT[0][0], ", ", AT[0][1], "], [", AT[1][0], ", ", AT[1][1], "]]")

    ck.check(C[0][0] == 19.0 and C[0][1] == 22.0 and C[1][0] == 43.0 and C[1][1] == 50.0,
             "matmul known values 19/22/43/50")
    print()

    # -------------------------------------------------- vector index
    print("--- Vector Index ---")
    var v1 = of4(0.1, 0.2, 0.3, 0.4)
    var v2 = of4(0.9, 0.8, 0.7, 0.6)
    var v3 = of4(0.5, 0.5, 0.5, 0.5)
    var index = VectorIndex(4)
    index.add("doc1", v1)
    index.add("doc2", v2)
    index.add("doc3", v3)
    var query = of4(0.15, 0.25, 0.35, 0.45)
    print("inputs: 3 docs (4-dim), query = [0.15, 0.25, 0.35, 0.45], k=3")
    var results = index.search(query, 3)
    for i in range(len(results)):
        print("  ", results[i][0], " score=", results[i][1])
    ck.check(results[0][0] == "doc1", "top result is doc1")
    ck.check(index.size() == 3, "index size == 3")
    var empty_index = VectorIndex(3)
    ck.check(len(empty_index.search(of3(1.0, 0.0, 0.0), 5)) == 0,
             "empty index search returns 0 results")
    print()

    # -------------------------------------------------- random projection
    print("--- Random Projection ---")
    var rp = RandomProjection(8, 3, seed=12345)
    var high_dim = of8(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    print("inputs: 8D vector [1,0,1,0,1,0,1,0], seed=12345, k=3")
    var low_dim = rp.project(high_dim)
    print_vec("8D -> 3D", low_dim)

    var rp_b = RandomProjection(5, 2, seed=999)
    var ones = of5(1.0, 1.0, 1.0, 1.0, 1.0)
    var p1 = rp_b.project(ones)
    var rp_c = RandomProjection(5, 2, seed=999)
    var p2 = rp_c.project(ones)
    ck.check(p1[0] == p2[0] and p1[1] == p2[1], "projection deterministic (same seed)")
    var rp_z = RandomProjection(4, 2, seed=7)
    var pz = rp_z.project(of4(0.0, 0.0, 0.0, 0.0))
    ck.check(pz[0] == 0.0 and pz[1] == 0.0, "projection of zero vector == zero (degenerate)")
    print()

    # -------------------------------------------------- quantization
    print("--- Scalar Quantization ---")
    var quantizer = ScalarQuantizer(4)
    var training = List[List[Float64]](capacity=3)
    training.append(copy_vec(v1))
    training.append(copy_vec(v2))
    training.append(copy_vec(v3))
    quantizer.fit(training)
    print("inputs: fit on 3 docs (4-dim); quantize doc1 = [0.1, 0.2, 0.3, 0.4]")
    var q1 = quantizer.quantize(v1)
    print("quantize(doc1) = [", q1[0], ", ", q1[1], ", ", q1[2], ", ", q1[3], "]")
    var dq1 = quantizer.dequantize(q1)
    print_vec("dequantize", dq1)
    ck.check(q1[0] == -128 and q1[1] == -128 and q1[2] == -128 and q1[3] == -128,
             "quantize(doc1) == [-128, -128, -128, -128]")
    var err_ok = True
    for j in range(4):
        var err = dq1[j] - v1[j]
        if err < 0.0:
            err = -err
        if err >= 0.05:
            err_ok = False
    ck.check(err_ok, "round-trip error < 0.05 per dim")
    var q_single = ScalarQuantizer(2)
    var single = List[List[Float64]](capacity=1)
    single.append(of2(0.0, 10.0))
    q_single.fit(single)
    var qsv = q_single.quantize(of2(0.0, 10.0))
    ck.check(qsv[0] == 0 and qsv[1] == 0, "zero-span quantize == 0 (degenerate)")
    print()

    # -------------------------------------------------- summary
    print("=== summary: ", ck.passed, " passed, ", ck.failed, " failed ===")
    if ck.failed == 0:
        print("DEMO: ALL CHECKS PASSED")
    else:
        print("DEMO: CHECKS FAILED")
    print()
    print("--- PARITY ---")
    print("dot4", bits(dot(a, b)))
    print("dot3_tail", bits(dot(of3(1.0, 2.0, 3.0), of3(4.0, 5.0, 6.0))))
    var a7 = of7(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    var b7 = of7(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    print("dot7_lane_tail", bits(dot(a7, b7)))
    print("dot_empty", bits(dot(List[Float64](), List[Float64]())))
    print("norm_340", bits(norm(of3(3.0, 4.0, 0.0))))
    print("cos_orth", bits(cosine_similarity(of2(1.0, 0.0), of2(0.0, 1.0))))
    print("cos_ident", bits(cosine_similarity(of3(1.0, 2.0, 3.0), of3(1.0, 2.0, 3.0))))
    print("cos_opp", bits(cosine_similarity(of2(1.0, 0.0), of2(-1.0, 0.0))))
    print("cos_zero", bits(cosine_similarity(of2(0.0, 0.0), of2(1.0, 0.0))))
    print("euc_34", bits(euclidean_distance(of2(0.0, 0.0), of2(3.0, 4.0))))
    var summed = add(a, b)
    for i in range(len(summed)):
        print("add4_e" + String(i), bits(summed[i]))
    var scaled = scale(of3(1.0, 2.0, 3.0), 2.0)
    for i in range(len(scaled)):
        print("scale3_e" + String(i), bits(scaled[i]))
    var nrm = normalize(of2(3.0, 4.0))
    for i in range(len(nrm)):
        print("norm34_e" + String(i), bits(nrm[i]))
    var nrm0 = normalize(of2(0.0, 0.0))
    for i in range(len(nrm0)):
        print("normzero_e" + String(i), bits(nrm0[i]))

    print("mm2_00", bits(C[0][0]))
    print("mm2_01", bits(C[0][1]))
    print("mm2_10", bits(C[1][0]))
    print("mm2_11", bits(C[1][1]))
    var I3 = List[List[Float64]](capacity=3)
    for i in range(3):
        var r = List[Float64](capacity=3)
        for j in range(3):
            if i == j:
                r.append(1.0)
            else:
                r.append(0.0)
        I3.append(r^)
    var A3 = List[List[Float64]](capacity=3)
    for i in range(3):
        var r = List[Float64](capacity=3)
        for j in range(3):
            r.append(Float64(i * 3 + j + 1))
        A3.append(r^)
    var R = matmul(A3, I3)
    print("mmI_00", bits(R[0][0]))
    print("mmI_12", bits(R[1][2]))
    print("mmI_22", bits(R[2][2]))

    print("rp8_e0", bits(low_dim[0]))
    print("rp8_e1", bits(low_dim[1]))
    print("rp8_e2", bits(low_dim[2]))
    print("rp5_e0", bits(p1[0]))
    print("rp5_e1", bits(p1[1]))
    print("rpzero_e0", bits(pz[0]))
    print("rpzero_e1", bits(pz[1]))

    for i in range(len(q1)):
        print("q_v1_e" + String(i), Int(q1[i]))
    for i in range(len(dq1)):
        print("dq_v1_e" + String(i), bits(dq1[i]))
    for i in range(len(qsv)):
        print("q_single_e" + String(i), Int(qsv[i]))
    var dqs = q_single.dequantize(qsv)
    for i in range(len(dqs)):
        print("dq_single_e" + String(i), bits(dqs[i]))

    for i in range(len(results)):
        print("idx_score_" + results[i][0], bits(results[i][1]))
    print("idx_empty_size", empty_index.size())
    print("idx_empty_results", len(empty_index.search(of3(1.0, 0.0, 0.0), 5)))
    print("--- END PARITY ---")
