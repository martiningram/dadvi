#!/usr/bin/env bash

echo If there are no error messages, the tests have passed.

python3 tests/test_core.py
python3 tests/conftest.py
python3 tests/pymc/test_jax_api.py