param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$planPath = Join-Path $projectRoot $Plan
$configuration = Get-Content -LiteralPath $planPath -Raw -Encoding utf8 | ConvertFrom-Json
$reviewPath = Join-Path $projectRoot $configuration.source_review
$review = Get-Content -LiteralPath $reviewPath -Raw -Encoding utf8 | ConvertFrom-Json
$planHash = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($review.status -ne 'SOURCE_PASS' -or $review.blocking_issues -ne 0 -or $review.plan_sha256 -ne $planHash) { throw 'Scheduled controller source review not passed.' }
$logDirectory = Join-Path $projectRoot $configuration.scheduler_logs
if (Test-Path -LiteralPath $logDirectory) { throw 'Scheduled invocation logs already exist; inspect the existing attempt.' }
New-Item -ItemType Directory -Path $logDirectory | Out-Null
$arguments = @('-u', $configuration.controller_entry, '--plan', $Plan)
if ($configuration.requires_toolkit) { $arguments += @('--toolkit','C:/Users/gb/Desktop/document/cadtrack/toolkit') }
$process = Start-Process -FilePath 'C:\Users\gb\AppData\Local\uv\cache\archive-v0\7oLWbeZ6v0oUodecKf4d1\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput (Join-Path $logDirectory 'stdout.log') -RedirectStandardError (Join-Path $logDirectory 'stderr.log')
$receipt = [ordered]@{ launcher_pid=$process.Id; exit_code=$process.ExitCode; completed_utc=[DateTime]::UtcNow.ToString('o'); plan_sha256=$planHash }
[System.IO.File]::WriteAllText((Join-Path $logDirectory 'exit.json'), (($receipt | ConvertTo-Json) + "`n"), [System.Text.UTF8Encoding]::new($false))
exit $process.ExitCode
