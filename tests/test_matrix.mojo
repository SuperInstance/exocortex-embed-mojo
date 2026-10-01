# test_matrix.mojo — Tests for matrix operations
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`
#   - builtin `assert(cond, msg)` uncallable in Mojo 1.2 -> fleet
#     self-reporting `Counters` pattern, 1:1 with original asserts.
#
# 15 assertions covering matmul, transpose, row, col.

from exocortex_embed.matrix import matmul, transpose, row, col


struct Counters(ImplicitlyCopyable):
    var total: Int
    var pass_count: Int
    var fail_count: Int

    def __init__(out self):
        self.total = 0
        self.pass_count = 0
        self.fail_count = 0

    def record(mut self, cond: Bool, test_name: String):
        self.total += 1
        if cond:
            self.pass_count += 1
            print("  PASS: ", test_name)
        else:
            self.fail_count += 1
            print("  FAIL: ", test_name)


def make_identity(n: Int) -> List[List[Float64]]:
    var M = List[List[Float64]](capacity=n)
    for i in range(n):
        var r = List[Float64](capacity=n)
        for j in range(n):
            if i == j:
                r.append(1.0)
            else:
                r.append(0.0)
        M.append(r^)
    return M^


def test_matmul_identity(mut c: Counters):
    var I = make_identity(3)
    var A = List[List[Float64]](capacity=3)
    for i in range(3):
        var r = List[Float64](capacity=3)
        for j in range(3):
            r.append(Float64(i * 3 + j + 1))
        A.append(r^)

    var result = matmul(A, I)
    # A × I = A
    c.record(result[0][0] == 1.0, "matmul identity [0][0]")
    c.record(result[1][2] == 6.0, "matmul identity [1][2]")
    c.record(result[2][2] == 9.0, "matmul identity [2][2]")


def test_matmul_known(mut c: Counters):
    # [[1, 2], [3, 4]] × [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    var A = List[List[Float64]](capacity=2)
    var r0 = List[Float64](capacity=2)
    r0.append(1.0); r0.append(2.0)
    var r1 = List[Float64](capacity=2)
    r1.append(3.0); r1.append(4.0)
    A.append(r0^); A.append(r1^)

    var B = List[List[Float64]](capacity=2)
    var c0 = List[Float64](capacity=2)
    c0.append(5.0); c0.append(6.0)
    var c1 = List[Float64](capacity=2)
    c1.append(7.0); c1.append(8.0)
    B.append(c0^); B.append(c1^)

    var C = matmul(A, B)
    c.record(C[0][0] == 19.0, "matmul known [0][0]")
    c.record(C[0][1] == 22.0, "matmul known [0][1]")
    c.record(C[1][0] == 43.0, "matmul known [1][0]")
    c.record(C[1][1] == 50.0, "matmul known [1][1]")


def test_transpose(mut c: Counters):
    var A = List[List[Float64]](capacity=2)
    var r0 = List[Float64](capacity=3)
    r0.append(1.0); r0.append(2.0); r0.append(3.0)
    var r1 = List[Float64](capacity=3)
    r1.append(4.0); r1.append(5.0); r1.append(6.0)
    A.append(r0^); A.append(r1^)

    var AT = transpose(A)
    # 2×3 → 3×2
    c.record(len(AT) == 3, "transpose rows")
    c.record(len(AT[0]) == 2, "transpose cols")
    c.record(AT[0][0] == 1.0, "transpose [0][0]")
    c.record(AT[2][1] == 6.0, "transpose [2][1]")


def test_row_col(mut c: Counters):
    var A = List[List[Float64]](capacity=2)
    var r0 = List[Float64](capacity=3)
    r0.append(1.0); r0.append(2.0); r0.append(3.0)
    var r1 = List[Float64](capacity=3)
    r1.append(4.0); r1.append(5.0); r1.append(6.0)
    A.append(r0^); A.append(r1^)

    var r = row(A, 1)
    c.record(r[0] == 4.0, "row[0]")
    c.record(r[2] == 6.0, "row[2]")

    var col_v = col(A, 1)
    c.record(col_v[0] == 2.0, "col[0]")
    c.record(col_v[1] == 5.0, "col[1]")


def main() raises:
    print("=== matrix tests ===")
    var c = Counters()
    test_matmul_identity(c)
    test_matmul_known(c)
    test_transpose(c)
    test_row_col(c)
    print("Results: ", c.pass_count, " passed, ", c.fail_count, " failed")
    if c.fail_count == 0:
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
