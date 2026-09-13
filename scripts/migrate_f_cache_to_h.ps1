$ErrorActionPreference='Stop'
$srcRoot='F:\RR_GID_CN_cache'
$dstRoot='H:\RR_GID_CN_data'
$items=Get-ChildItem -LiteralPath $srcRoot -Directory -Force
$rows=@()
foreach($src in $items){
  $class=if($src.Name -match 'max20'){'active'}else{'archived'}
  $dst=Join-Path (Join-Path $dstRoot $class) ('f_cache\'+$src.Name)
  New-Item -ItemType Directory -Force -Path $dst | Out-Null
  $before=Get-ChildItem -LiteralPath $src.FullName -File -Recurse -Force | Measure-Object Length -Sum
  & robocopy $src.FullName $dst /E /COPY:DAT /DCOPY:T /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
  if($LASTEXITCODE -gt 7){throw ('robocopy failed: '+$src.FullName)}
  $after=Get-ChildItem -LiteralPath $dst -File -Recurse -Force | Measure-Object Length -Sum
  if($before.Count -ne $after.Count -or $before.Sum -ne $after.Sum){throw ('verification failed: '+$src.FullName)}
  Remove-Item -LiteralPath $src.FullName -Recurse -Force
  $rows += [pscustomobject]@{Source=$src.FullName;Destination=$dst;Class=$class;Files=$before.Count;Bytes=[int64]$before.Sum;Verification='robocopy then recursive file count and byte sum'}
}
$rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $dstRoot 'manifests\f_cache_migration_20260907.json') -Encoding UTF8
$rows | Format-Table -AutoSize
