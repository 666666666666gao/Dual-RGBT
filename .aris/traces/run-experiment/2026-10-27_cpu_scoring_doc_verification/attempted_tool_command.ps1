$ErrorActionPreference = 'Stop'
$traceDir = 'C:\Users\gb\projects\Dual-RGBT\.aris\traces\run-experiment\2026-10-27_cpu_scoring_doc_verification'
[System.IO.Directory]::CreateDirectory($traceDir) | Out-Null
$docPath = 'C:\Users\gb\projects\Dual-RGBT\docs\CPU_SCORING_ENV_20261027.md'
$ledgerPath = 'C:\Users\gb\projects\Dual-RGBT\.aris\compute\ssh-server25.md'
$skillPath = 'C:\Users\gb\.codex\skills\run-experiment\SKILL.md'
$contractPath = 'C:\Users\gb\.codex\skills\shared-references\compute-env-contract.md'
$docText = [System.IO.File]::ReadAllText($docPath)
$commandText = [regex]::Match($docText, '(?s)```powershell\r?\n(.*?)\r?\n```').Groups[1].Value
$encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $traceDir 'documented_command.txt'), $commandText, $encoding)
$scriptPath = Join-Path $traceDir 'documented_invocation.ps1'
[System.IO.File]::WriteAllText($scriptPath, $commandText + [Environment]::NewLine + 'exit $LASTEXITCODE' + [Environment]::NewLine, $encoding)
Copy-Item -LiteralPath $docPath -Destination (Join-Path $traceDir 'CPU_SCORING_ENV_20261027.source.md')
Copy-Item -LiteralPath $ledgerPath -Destination (Join-Path $traceDir 'ssh-server25.source.md')
$sources = @($docPath, $ledgerPath, $skillPath, $contractPath) | ForEach-Object { [ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()} }
$sourceReceipt = [ordered]@{created_utc=[DateTime]::UtcNow.ToString('o');agent_task='/root/verify_cpu_scoring_env_document_20261027';requested_model='Astra';requested_reasoning='max';actual_model='UNATTESTED';review_classification='same-family/provisional';source_files=$sources;documented_command=$commandText;invocation_script_sha256=(Get-FileHash -LiteralPath $scriptPath -Algorithm SHA256).Hash.ToLowerInvariant();ledger_declared_canonical_spec_sha256='5e574a086db1d396a0be0f1852abc0d8a77809bf882f567ff5684a5948e9ea14';scope='CPU-only documented verification; no build, NN, checkpoint, GPU queries, training control, or source edits';powershell_version=$PSVersionTable.PSVersion.ToString()}
[System.IO.File]::WriteAllText((Join-Path $traceDir 'source_receipt.json'), ($sourceReceipt | ConvertTo-Json -Depth 8), $encoding)
$startedUtc = [DateTime]::UtcNow
$timer = [Diagnostics.Stopwatch]::StartNew()
$process = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') -ArgumentList @('-NoLogo','-NoProfile','-NonInteractive','-File',$scriptPath) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $traceDir 'verification.stdout.txt') -RedirectStandardError (Join-Path $traceDir 'verification.stderr.txt') -Wait -PassThru
$timer.Stop()
$nativeReceipt = [ordered]@{started_utc=$startedUtc.ToString('o');finished_utc=[DateTime]::UtcNow.ToString('o');elapsed_seconds=$timer.Elapsed.TotalSeconds;process_id=$process.Id;native_exit_code=$process.ExitCode;documented_invocation_executions=1;verbatim_command_file='documented_command.txt';powershell_exit_propagation='exit $LASTEXITCODE';stdout_file='verification.stdout.txt';stderr_file='verification.stderr.txt';stdout_sha256=(Get-FileHash -LiteralPath (Join-Path $traceDir 'verification.stdout.txt') -Algorithm SHA256).Hash.ToLowerInvariant();stderr_sha256=(Get-FileHash -LiteralPath (Join-Path $traceDir 'verification.stderr.txt') -Algorithm SHA256).Hash.ToLowerInvariant()}
[System.IO.File]::WriteAllText((Join-Path $traceDir 'native_execution_receipt.json'), ($nativeReceipt | ConvertTo-Json -Depth 6), $encoding)
$nativeReceipt | ConvertTo-Json -Depth 6
Get-Content -LiteralPath (Join-Path $traceDir 'verification.stdout.txt') -Raw
Get-Content -LiteralPath (Join-Path $traceDir 'verification.stderr.txt') -Raw
