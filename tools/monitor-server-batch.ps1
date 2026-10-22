[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JobListPath,
    [string]$ApiBaseUrl = "http://127.0.0.1:18000",
    [string]$OutputDirectory = "",
    [ValidateRange(5, 300)]
    [int]$PollSeconds = 20
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resolvedJobList = (Resolve-Path -LiteralPath $JobListPath).Path
$jobListText = [System.IO.File]::ReadAllText(
    $resolvedJobList,
    [System.Text.Encoding]::UTF8
)
$parsedJobs = $jobListText | ConvertFrom-Json
$jobs = @($parsedJobs | ForEach-Object { $_ })
if ($jobs.Count -eq 0) {
    throw "The job list is empty"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $fileName = [System.IO.Path]::GetFileNameWithoutExtension($resolvedJobList)
    if ($fileName.EndsWith("-jobs")) {
        $fileName = $fileName.Substring(0, $fileName.Length - 5)
    }
    $OutputDirectory = Join-Path (
        [System.IO.Path]::GetDirectoryName($resolvedJobList)
    ) "$fileName-results"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$healthText = & curl.exe --silent --show-error --fail-with-body `
    --connect-timeout 15 --max-time 30 "$ApiBaseUrl/health"
if ($LASTEXITCODE -ne 0) {
    throw "API health check failed. Verify the SSH tunnel."
}
$health = $healthText | ConvertFrom-Json
if ($health.status -ne "ok") {
    throw "API health status is not ok"
}

$utf8 = New-Object System.Text.UTF8Encoding($false)
$seen = @{}

while ($seen.Count -lt $jobs.Count) {
    foreach ($job in $jobs) {
        $labelProperty = $job.PSObject.Properties["specification"]
        if ($null -eq $labelProperty) {
            $labelProperty = $job.PSObject.Properties["key"]
        }
        $label = if ($null -ne $labelProperty) {
            [string]$labelProperty.Value
        } else {
            [string]$job.job_id
        }
        if ($seen.ContainsKey($label)) {
            continue
        }

        $safeLabel = [System.Text.RegularExpressions.Regex]::Replace(
            $label,
            "[^A-Za-z0-9_.-]",
            "_"
        )
        $temporaryFile = Join-Path $env:TEMP "server-job-$safeLabel.json"
        try {
            & curl.exe --silent --show-error `
                --connect-timeout 15 `
                --max-time 30 `
                --output $temporaryFile `
                "$ApiBaseUrl/api/v1/checks/$($job.job_id)"
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Temporary API error for $label"
                continue
            }

            $json = [System.IO.File]::ReadAllText(
                $temporaryFile,
                [System.Text.Encoding]::UTF8
            )
            $response = $json | ConvertFrom-Json
        }
        finally {
            if (Test-Path -LiteralPath $temporaryFile) {
                Remove-Item -LiteralPath $temporaryFile -Force
            }
        }

        if ($response.status -notin @("COMPLETED", "FAILED")) {
            continue
        }

        $resultPath = Join-Path $OutputDirectory "$safeLabel.json"
        [System.IO.File]::WriteAllText($resultPath, $json, $utf8)

        $totalSeconds = $response.processing_time_seconds
        $checkSeconds = $null
        $decision = ""
        $comment = ""
        if ($null -ne $response.result) {
            $decision = [string]$response.result.decision
            $comment = [string]$response.result.comment_1c
            $checkSeconds = $response.result.metadata.processing_time_seconds
        } elseif ($null -ne $response.error) {
            $comment = [string]$response.error.message
        }
        $queueSeconds = if (
            $null -ne $totalSeconds -and $null -ne $checkSeconds
        ) {
            [math]::Round(
                [double]$totalSeconds - [double]$checkSeconds,
                2
            )
        } else {
            $null
        }

        Write-Host ""
        Write-Host ("{0}: finished" -f $label) -ForegroundColor Green
        [pscustomobject]@{
            specification = $label
            status = $response.status
            decision = $decision
            check_seconds = $checkSeconds
            queue_wait_seconds = $queueSeconds
            total_seconds = $totalSeconds
            comment = $comment
        } | Format-List
        $seen[$label] = $true
    }

    Write-Host ("Completed: {0} of {1}. Next poll in {2}s." -f $seen.Count, $jobs.Count, $PollSeconds)
    if ($seen.Count -lt $jobs.Count) {
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-Host ("All {0} jobs are finished." -f $jobs.Count)
Write-Host ("Results: {0}" -f $OutputDirectory)
