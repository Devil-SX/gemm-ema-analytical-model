# GEMM EMA Repro
Timeloop: https://github.com/NVlabs/timeloop
Orojenesis: https://timeloop.csail.mit.edu/orojenesis
Artifact: https://zenodo.org/records/12600121/files/orojenesis.zip?download=1

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

```bash
./scripts/fetch_orojenesis_artifact.sh
python3 scripts/run_orojenesis_mm_same_flops.py --check-env
python3 scripts/run_orojenesis_mm_same_flops.py
python3 scripts/make_validation_figure.py
```
