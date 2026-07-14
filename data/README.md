# Datasets

Raw data and generated `processed/` files are intentionally excluded from Git. Download each dataset from its
official source and run `prepare_data.py`; see the repository root `README.md` for commands and expected files.

Custom datasets prepared from a JSON specification are written to `data/custom/<name>/processed/`. They can also
remain outside the repository and be selected with `main.py --data-dir /path/to/<name>`.

Do not commit WADI files: access is granted by SUTD under its own terms of use.
