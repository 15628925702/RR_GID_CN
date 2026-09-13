param(
    [switch]$LaunchP7
)

$ErrorActionPreference = "Continue"
$root = "G:\\0-newResearch\\4.RR_GID_CN"
$python = "F:\\Python\\python.exe"
$watchRoot = "H:\\RR_GID_CN_data\\active\\repaired_20260912\\pipeline_watchdog"
$syntheticDir = "H:\\RR_GID_CN_data\\active\\repaired_20260912\\e1_validated_fixed_order14_j2_200"
$syntheticRows = Join-Path $syntheticDir "rows.jsonl"
$syntheticConfig = Join-Path $root "configs\\paper\\synthetic_main_validated_fixed_order14_j2_200_h.yaml"
$syntheticPrepared = Join-Path $root "experiments\\paper\\oracle_artifact.pkl"
$syntheticCertificate = "H:\\RR_GID_CN_data\\active\\repaired_20260911\\certificates\\fixed_cached_equivalence_order14.json"
$syntheticAudit = Join-Path $syntheticDir "AUDIT_synthetic_order14.json"
$reuseDir = "H:\\RR_GID_CN_data\\active\\repaired_20260912\\p7_reuse_order14_t50"
$reuseRows = Join-Path $reuseDir "rows.jsonl"
$reuseConfig = Join-Path $root "configs\\paper\\reuse_order14_t50_h.yaml"
$reusePrepared = Join-Path $root "experiments\\paper\\oracle_artifact.pkl"
$reuseAudit = Join-Path $reuseDir "AUDIT_reuse_order14_t50.json"

New-Item -ItemType Directory -Force -Path $watchRoot | Out-Null
$logPath = Join-Path $watchRoot "watchdog.log"

function Log([string]$Message) {
    $line = "$(Get-Date -Format o) $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function RowCount([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    return [int](Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

function IsRunning([string]$Pattern) {
    return $null -ne (Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -eq "python.exe" -and $_.CommandLine -match $Pattern
    } | Select-Object -First 1)
}

function StartSynthetic {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = Join-Path $watchRoot "synthetic_$stamp.out.log"
    $stderr = Join-Path $watchRoot "synthetic_$stamp.err.log"
    $args = @(
        "scripts/run_synthetic_main.py",
        "--config", $syntheticConfig,
        "--prepared", $syntheticPrepared,
        "--resume", "--profile"
    )
    $p = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    Log "started synthetic pid=$($p.Id) stdout=$stdout stderr=$stderr"
}

Log "watchdog started launchP7=$LaunchP7"
while ((RowCount $syntheticRows) -lt 4000) {
    $n = RowCount $syntheticRows
    if (-not (IsRunning "run_synthetic_main\.py.*synthetic_main_validated_fixed_order14_j2_200_h\.yaml")) {
        StartSynthetic
    }
    Log "synthetic rows=$n/4000"
    Start-Sleep -Seconds 60
}

# Give the writer a moment to flush its final line before auditing.
Start-Sleep -Seconds 10
Log "synthetic grid reached 4000; running fail-closed audit"
& $python "scripts/audit_repaired_synthetic.py" --rows $syntheticRows --config $syntheticConfig --certificate $syntheticCertificate --out $syntheticAudit *> (Join-Path $watchRoot "synthetic_audit.log")
$auditCode = $LASTEXITCODE
Log "synthetic audit exit=$auditCode report=$syntheticAudit"
if ($auditCode -ne 0) {
    Log "synthetic audit rejected; stopping watchdog before P7"
    exit $auditCode
}

if (-not $LaunchP7) {
    Log "LaunchP7 not requested; watchdog complete"
    exit 0
}

New-Item -ItemType Directory -Force -Path $reuseDir | Out-Null
while ((RowCount $reuseRows) -lt 500) {
    $n = RowCount $reuseRows
    if (-not (IsRunning "run_reuse\.py.*reuse_order14_t50_h\.yaml")) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $stdout = Join-Path $watchRoot "reuse_$stamp.out.log"
        $stderr = Join-Path $watchRoot "reuse_$stamp.err.log"
        $args = @(
            "scripts/run_reuse.py",
            "--config", $reuseConfig,
            "--prepared", $reusePrepared,
            "--resume", "--profile"
        )
        $p = Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        Log "started reuse pid=$($p.Id) stdout=$stdout stderr=$stderr"
    }
    Log "reuse rows=$n/500"
    Start-Sleep -Seconds 60
}

Start-Sleep -Seconds 10
Log "reuse grid reached 500; running fail-closed audit"
& $python "scripts/audit_reuse_order14_t50.py" --rows $reuseRows --config $reuseConfig --out $reuseAudit *> (Join-Path $watchRoot "reuse_audit.log")
$reuseCode = $LASTEXITCODE
Log "reuse audit exit=$reuseCode report=$reuseAudit"
exit $reuseCode
