# Upload guide

This folder is a complete repository-ready package. The published repository is public and includes `models/frozen_rf31_bundle.joblib`; anyone can download the frozen model.

## Recommended route

1. Open the target GitHub repository in GitHub Desktop or a Git client and clone it to a fresh local folder.
2. Copy the **contents** of this package into that checkout, preserving the `app/`, `models/`, `scripts/`, and `docs/` directories.
3. If updating an older flat version of this project, remove the superseded root-level files (`predictor.py`, `runtime.py`, `serve.py`, `build_bundle.py`, `verify_web_model.py`, `self_test.py`, `frozen_rf31_bundle.joblib`, `MODEL_SHA256.txt`, `VERSION.json`, `bundle_audit.json`, `web_verification.json`) and old root `static/` directory from the checkout before committing. The current root-level launchers and `requirements.txt` should be replaced with those in this package.
4. Review the staged changes, commit them, and push to `main`.
5. Verify GitHub shows the four directories, `README.md`, `requirements.txt`, and the Windows launchers at the repository root.

The GitHub web uploader requires entering each target folder before selecting its files; uploading a ZIP as a repository file does **not** unpack it. The ZIP supplied alongside this folder is for transfer and backup only.

## Check before committing

```powershell
Get-FileHash .\models\frozen_rf31_bundle.joblib -Algorithm SHA256
.\.venv\Scripts\python.exe -m scripts.self_test
```

The model hash should match `models/MODEL_SHA256.txt`. `SHA256SUMS.txt` lists the other package files. Do not add patient workbooks, local `work/` and `outputs/` directories, access tokens, or generated virtual environments.
