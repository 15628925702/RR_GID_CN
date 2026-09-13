$ErrorActionPreference = 'Stop'
$base = 'G:\0-newResearch\4.RR_GID_CN\results\validation_20260906'
$h = 'H:\RR_GID_CN_data\archived\validation_20260906'
$rows = @()
$dirs = Get-ChildItem -LiteralPath $base -Directory -Recurse -Force | Where-Object Name -eq 'basis_cache'
foreach ($src in $dirs) {
  $run = $src.Parent.Name
  $dst = Join-Path (Join-Path $h $run) 'basis_cache'
  $b = Get-ChildItem -LiteralPath $src.FullName -File -Recurse -Force | Measure-Object Length -Sum
  $d = Get-ChildItem -LiteralPath $dst -File -Recurse -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum
  if ($b.Count -ne $d.Count -or $b.Sum -ne $d.Sum) {
    & robocopy $src.FullName $dst /E /COPY:DAT /DCOPY:T /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -gt 7) { throw 'copy failed' }
    $d = Get-ChildItem -LiteralPath $dst -File -Recurse -Force | Measure-Object Length -Sum
  }
  if ($b.Count -ne $d.Count -or $b.Sum -ne $d.Sum) { throw 'verify failed' }
  Remove-Item -LiteralPath $src.FullName -Recurse -Force
  $rows += [pscustomobject]@{Run=$run;Source=$src.FullName;Destination=$dst;Files=$b.Count;Bytes=[int64]$b.Sum;Verification='recursive count+bytes'}
}
$rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath 'H:\RR_GID_CN_data\manifests\migration_20260907.json' -Encoding UTF8
$rows | Format-Table -AutoSize
