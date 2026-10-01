# test_index.mojo — Tests for vector index
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`
#   - builtin `assert(cond, msg)` uncallable in Mojo 1.2 -> fleet
#     self-reporting `Counters` pattern, 1:1 with original asserts.
#
# 14 assertions covering add, search, remove, size.
#
# BOOKED (not fixed, fleet rule — semantic): in test_remove the original has
# `v2.append(0.0); v1.append(1.0)` — almost certainly meant `v2.append(1.0)`.
# As-is, v2 holds only ONE element, so the post-remove search computes
# dot(query[2], entry[1]) and reads out of bounds. Originals preserved in git.

from exocortex_embed.index import VectorIndex


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


def test_add_and_size(mut c: Counters):
    var index = VectorIndex(3)
    c.record(index.size() == 0, "initial size")

    var v = List[Float64](capacity=3)
    v.append(1.0); v.append(0.0); v.append(0.0)
    index.add("a", v)
    c.record(index.size() == 1, "size after add")

    var v2 = List[Float64](capacity=3)
    v2.append(0.0); v2.append(1.0); v2.append(0.0)
    index.add("b", v2)
    c.record(index.size() == 2, "size after second add")


def test_search_top_k(mut c: Counters):
    var index = VectorIndex(3)

    var v1 = List[Float64](capacity=3)
    v1.append(1.0); v1.append(0.0); v1.append(0.0)
    var v2 = List[Float64](capacity=3)
    v2.append(0.0); v2.append(1.0); v2.append(0.0)
    var v3 = List[Float64](capacity=3)
    v3.append(0.0); v3.append(0.0); v3.append(1.0)

    index.add("x", v1)
    index.add("y", v2)
    index.add("z", v3)

    var query = List[Float64](capacity=3)
    query.append(0.9); query.append(0.1); query.append(0.0)

    var results = index.search(query, 2)
    c.record(len(results) == 2, "search returns k results")
    # "x" should be top result (closest to query direction)
    c.record(results[0][0] == "x", "top result is x")


def test_search_empty(mut c: Counters):
    var index = VectorIndex(3)
    var query = List[Float64](capacity=3)
    query.append(1.0); query.append(0.0); query.append(0.0)

    var results = index.search(query, 5)
    c.record(len(results) == 0, "empty index returns empty")


def test_remove(mut c: Counters):
    var index = VectorIndex(2)

    var v1 = List[Float64](capacity=2)
    v1.append(1.0); v1.append(0.0)
    var v2 = List[Float64](capacity=2)
    # ORIGINAL (booked, unfixed): `v2.append(0.0); v1.append(1.0)` — v2/v1 typo.
    # Left exactly as authored; see header note.
    v2.append(0.0); v1.append(1.0)

    index.add("a", v1)
    index.add("b", v2)
    c.record(index.size() == 2, "size before remove")

    index.remove("a")
    c.record(index.size() == 1, "size after remove")

    # Search should only find "b"
    var query = List[Float64](capacity=2)
    query.append(1.0); query.append(0.0)
    var results = index.search(query, 5)
    c.record(len(results) == 1, "search after remove")
    c.record(results[0][0] == "b", "remaining is b")


def test_search_ordering(mut c: Counters):
    var index = VectorIndex(2)

    var v1 = List[Float64](capacity=2)
    v1.append(1.0); v1.append(0.0)
    var v2 = List[Float64](capacity=2)
    v2.append(0.8); v2.append(0.6)
    var v3 = List[Float64](capacity=2)
    v3.append(0.0); v3.append(1.0)

    index.add("right", v1)
    index.add("diag", v2)
    index.add("up", v3)

    var query = List[Float64](capacity=2)
    query.append(1.0); query.append(0.0)

    var results = index.search(query, 3)
    # Should be ordered: right > diag > up
    c.record(results[0][0] == "right", "first is right")
    c.record(results[1][0] == "diag", "second is diag")
    c.record(results[2][0] == "up", "third is up")


def main() raises:
    print("=== index tests ===")
    var c = Counters()
    test_add_and_size(c)
    test_search_top_k(c)
    test_search_empty(c)
    test_remove(c)
    test_search_ordering(c)
    print("Results: ", c.pass_count, " passed, ", c.fail_count, " failed")
    if c.fail_count == 0:
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
