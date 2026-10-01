# exocortex-embed-mojo — on-box build/test/demo/parity contract
#
# This repo has NO pixi manifest; the proven toolchain on this box is
# quilt-mojo-lab's pixi environment via MODULAR_HOME + PATH (see
# docs/USERMANUAL.md). Flags BEFORE filename; -D ASSERT=all per fleet rule.

MODULAR_HOME := $(HOME)/projects/quilt-mojo-lab/.pixi/envs/default/share/max
export MODULAR_HOME
export PATH := $(HOME)/projects/quilt-mojo-lab/.pixi/envs/default/bin:$(PATH)

TESTS := vector matrix random_proj index quantize
BUILD := build

.PHONY: all test demo parity clean

all: test demo parity

test:
	@mkdir -p $(BUILD)
	@for t in $(TESTS); do \
		echo "=== build test_$$t ==="; \
		mojo build -I src -D ASSERT=all tests/test_$$t.mojo -o $(BUILD)/test_$$t || exit 1; \
		$(BUILD)/test_$$t || exit 1; \
	done

demo:
	@mkdir -p $(BUILD)
	mojo build -I src -D ASSERT=all examples/demo.mojo -o $(BUILD)/demo
	$(BUILD)/demo

parity: demo
	python3 python/parity_check.py --mojo $(BUILD)/demo

clean:
	rm -rf $(BUILD)
