# -*- Python -*-

# LIT configuration to run unit tests as part of the test suite

import os

import lit.formats

config.name = "amd-codeobj-to-llvm-Unit"
config.suffixes = []

# The root path where tests should be run.
config.test_exec_root = config.amd_codeobj_gtest_lit_exedir

# The root path where tests are located.
config.test_source_root = config.test_exec_root

# Extract gTests as LIT tests.
config.test_format = lit.formats.GoogleTest(config.llvm_build_mode, "")

# Propagate directory env variables.
if "TMP" in os.environ:
    config.environment["TMP"] = os.environ["TMP"]
if "TEMP" in os.environ:
    config.environment["TEMP"] = os.environ["TEMP"]
if "HOME" in os.environ:
    config.environment["HOME"] = os.environ["HOME"]

# Propagate sanitizer options.
for var in (
    "ASAN_SYMBOLIZER_PATH",
    "HWASAN_SYMBOLIZER_PATH",
    "MSAN_SYMBOLIZER_PATH",
    "TSAN_SYMBOLIZER_PATH",
    "UBSAN_SYMBOLIZER_PATH",
    "ASAN_OPTIONS",
    "HWASAN_OPTIONS",
    "MSAN_OPTIONS",
    "TSAN_OPTIONS",
    "UBSAN_OPTIONS",
):
    if var in os.environ:
        config.environment[var] = os.environ[var]
