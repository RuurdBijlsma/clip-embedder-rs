Write-Host " ----- [ ruff format ] ----- "
uvx ruff format .\general_export_onnx.py

Write-Host " ----- [ ruff check --fix ] ----- "
uvx ruff check --fix .\general_export_onnx.py