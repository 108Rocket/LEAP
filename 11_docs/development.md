# Development

```bash
python -m pip install -e ".[test]" --no-deps
python 09_tools/run_checks.py --cpu
python -m unittest discover -s 08_tests/tests/rq1 -p test_external_interfaces.py
```

Use `python 09_tools/run_checks.py` for the full suite on a supported CUDA host.
The [test directory](../08_tests/tests/README.md) groups checks by RQ1--RQ5 and shared
support. Run one group with `python -m pytest 08_tests/tests/rq1` from the release root.
The [validation record](validation.md) states what was checked for this release.
External-interface tests need only Python and Git, without external models,
checkpoints, datasets or GPU libraries.
The `run_checks.py` entry saves console output, the selected test configuration
and JUnit results under `14_logs/tests/`. Direct pytest/unittest commands retain
their standard console-only behavior unless output options are supplied.

Keep shared numerical code in `01_LEAP/leap/common` and distance-specific code in its
distance directory. The [architecture guide](architecture.md) describes the
call flow. Import implementation modules from their owning subpackages.

[Overview](../README.md)
