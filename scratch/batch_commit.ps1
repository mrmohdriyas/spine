$files = git ls-files -o --exclude-standard
$maxBatchSizeBytes = 50 * 1024 * 1024 # 50 MB threshold per commit
$maxBatchCount = 100 # Fallback threshold to ensure the git add command doesn't get too long
$batch = @()
$currentBatchSize = 0

# Auto-detect the next commit number from git log
$lastCommitMsg = (git log -1 --pretty=%B) -join "`n"
$commitNum = 1
if ($lastCommitMsg -match "Batch commit (\d+)") {
    $commitNum = [int]$matches[1] + 1
}

foreach ($file in $files) {
    if ([string]::IsNullOrWhiteSpace($file)) { continue }
    
    $fileInfo = Get-Item $file -ErrorAction SilentlyContinue
    $fileSize = if ($fileInfo) { $fileInfo.Length } else { 0 }
    
    # If adding this file exceeds the 50MB limit AND we already have files in the batch,
    # commit the current batch first.
    if ($currentBatchSize + $fileSize -ge $maxBatchSizeBytes -and $batch.Count -gt 0) {
        Write-Host "Committing batch $commitNum with $($batch.Count) files (Size: $([math]::Round($currentBatchSize / 1MB, 2)) MB)..."
        git add $batch
        git commit -m "Batch commit $commitNum"
        
        Write-Host "Pushing batch $commitNum..."
        git push origin main
        
        $batch = @()
        $currentBatchSize = 0
        $commitNum++
    }

    $batch += $file
    $currentBatchSize += $fileSize

    # Fallback to cap by file count if there are many tiny files
    if ($batch.Count -ge $maxBatchCount) {
        Write-Host "Committing batch $commitNum with $($batch.Count) files (Size: $([math]::Round($currentBatchSize / 1MB, 2)) MB)..."
        git add $batch
        git commit -m "Batch commit $commitNum"
        
        Write-Host "Pushing batch $commitNum..."
        git push origin main
        
        $batch = @()
        $currentBatchSize = 0
        $commitNum++
    }
}

if ($batch.Count -gt 0) {
    Write-Host "Committing final batch $commitNum with $($batch.Count) files (Size: $([math]::Round($currentBatchSize / 1MB, 2)) MB)..."
    git add $batch
    git commit -m "Batch commit $commitNum"
    
    Write-Host "Pushing final batch $commitNum..."
    git push origin main
}

Write-Host "Batch push completed successfully!"
