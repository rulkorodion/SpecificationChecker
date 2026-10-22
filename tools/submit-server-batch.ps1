[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "http://127.0.0.1:18000",
    [string]$BatchName = "server-batch-5-55-20260806",
    [int[]]$SpecificationIds = @(
        5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        18, 19, 20, 21, 22, 25, 26, 27, 28, 29,
        31, 33, 40, 41, 42, 43, 44, 45,
        50, 51, 52, 53, 54, 55
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$testdataRoot = Join-Path $projectRoot "testdata\specifications"
$documentsRoot = Join-Path $testdataRoot "documents"
$referenceRoot = Join-Path $testdataRoot "reference"
$runtimeRoot = Join-Path $projectRoot "runtime"
$jobListPath = Join-Path $runtimeRoot "$BatchName-jobs.json"
$utf8 = New-Object System.Text.UTF8Encoding($false)
$specificationWord = [System.Text.Encoding]::UTF8.GetString(
    [System.Convert]::FromBase64String("0KHQv9C10YbQuNGE0LjQutCw0YbQuNGP")
)

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is not available"
}

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

$healthText = & curl.exe --silent --show-error --fail-with-body `
    --connect-timeout 15 --max-time 30 "$ApiBaseUrl/health"
if ($LASTEXITCODE -ne 0) {
    throw "API health check failed. Verify the SSH tunnel."
}
$health = $healthText | ConvertFrom-Json
if ($health.status -ne "ok" -or -not $health.worker_registered) {
    throw "API is not healthy or the worker is not registered"
}

$jobs = @()
if (Test-Path -LiteralPath $jobListPath) {
    $savedText = [System.IO.File]::ReadAllText(
        $jobListPath,
        [System.Text.Encoding]::UTF8
    )
    if (-not [string]::IsNullOrWhiteSpace($savedText)) {
        $parsedJobs = $savedText | ConvertFrom-Json
        $jobs = @($parsedJobs | ForEach-Object { $_ })
    }
}

$allDocuments = @(Get-ChildItem -LiteralPath $documentsRoot -File)

foreach ($number in $SpecificationIds) {
    $submittedNumbers = @(
        $jobs | ForEach-Object { [int]$_.specification }
    )
    if ($submittedNumbers -contains $number) {
        Write-Host ("Specification {0}: already submitted, skipping" -f $number)
        continue
    }

    # Keep the script itself ASCII-compatible for Windows PowerShell 5.1.
    $expectedBaseName = "{0} {1}" -f $specificationWord, $number
    $matches = @(
        $allDocuments |
            Where-Object { $_.BaseName -eq $expectedBaseName }
    )
    if ($matches.Count -ne 1) {
        throw ("Specification {0}: expected one document, found {1}" -f $number, $matches.Count)
    }

    $document = $matches[0].FullName
    $reference = Join-Path $referenceRoot ("specification_{0}.json" -f $number)
    if (-not (Test-Path -LiteralPath $reference)) {
        throw ("Reference is missing: {0}" -f $reference)
    }

    Write-Host ("Submitting specification {0}: {1}" -f $number, $matches[0].Name)
    $responseText = & curl.exe --silent --show-error --fail-with-body `
        --connect-timeout 15 `
        --max-time 120 `
        --form ("document=@{0}" -f $document) `
        --form ("reference=<{0}" -f $reference) `
        --form-string ("external_id={0}-specification-{1}" -f $BatchName, $number) `
        "$ApiBaseUrl/api/v1/checks"
    if ($LASTEXITCODE -ne 0) {
        throw ("Specification {0}: upload failed. Rerun this script to resume." -f $number)
    }

    $response = $responseText | ConvertFrom-Json
    $jobs += [pscustomobject]@{
        specification = $number
        document = $matches[0].Name
        job_id = $response.job_id
        status = $response.status
    }

    $serialized = $jobs | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText($jobListPath, $serialized, $utf8)
    Write-Host ("  job_id={0}, status={1}" -f $response.job_id, $response.status)
}

$jobs | Format-Table specification, document, job_id, status -AutoSize
Write-Host ("Submitted jobs: {0} of {1}" -f $jobs.Count, $SpecificationIds.Count)
Write-Host ("Job list: {0}" -f $jobListPath)
