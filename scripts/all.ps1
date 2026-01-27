<#
.SYNOPSIS
Rust quality checks.
#>
$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSVersion.Major -lt 5) { throw "PowerShell 5.1+ required." }

# Environment Setup
$env:CARGO_TERM_COLOR = "always"
$env:RUSTFLAGS = "-Dwarnings"

Write-Host "=== 1. Validating Environment ===" -ForegroundColor Cyan

# Check for Cargo first
if (-not (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
    throw "Cargo not found. Please install Rust from https://rustup.rs"
}

# Fast check for components - only runs rustup if the command fails
try {
    cargo fmt --version > $null 2>&1
    if ($LASTEXITCODE -ne 0) { rustup component add rustfmt }

    cargo clippy --version > $null 2>&1
    if ($LASTEXITCODE -ne 0) { rustup component add clippy }
} catch {
    Write-Host "Warning: Could not verify components via rustup." -ForegroundColor Yellow
}

try {
    # - Format Check (Fail fast - takes < 1 second)
    Write-Host "`n=== 2. Checking Format ===" -ForegroundColor Cyan
    cargo fmt --all
    cargo fmt --all -- --check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Formatting issues found. Run 'cargo fmt --all' to fix." -ForegroundColor Red
        exit 1
    }

    # - Combined Clippy & Build (Using Release profile)
    # By using --release here, we cache the builds for the tests later.
    Write-Host "`n=== 3. Running Clippy (Release Profile) ===" -ForegroundColor Cyan
    cargo clippy --release --all-targets --all-features -- -D clippy::pedantic -D clippy::nursery
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # - Optimized Testing
    # This reuses the artifacts from the Clippy step above.
    Write-Host "`n=== 4. Running All Tests (Release Profile) ===" -ForegroundColor Cyan
    cargo test --release --workspace --all-features -- --nocapture
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # - Documentation check
    Write-Host "`n=== 5. Checking Documentation ===" -ForegroundColor Cyan
    cargo doc --no-deps --document-private-items --all-features

    Write-Host "`n=== Success: All checks passed! ===" -ForegroundColor Green
}
catch {
    Write-Host "`n[!] SCRIPT ERROR: $_" -ForegroundColor Red
    exit 1
}