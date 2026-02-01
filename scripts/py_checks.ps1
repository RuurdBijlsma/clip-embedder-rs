Write-Host " ----- [ ruff format ] ----- "
uvx ruff format .\pull_onnx.py

Write-Host " ----- [ ruff check --fix ] ----- "
uvx ruff check --select ALL --fix .\pull_onnx.py