# Auto detect proxy and push to git
# Time complexity O(n), space complexity O(1)

$ports = @(7890, 10809, 1080)
$proxyFound = $false

foreach ($p in $ports) {
    $conn = (Test-NetConnection 127.0.0.1 -Port $p -InformationLevel Quiet)
    if ($conn) {
        git config --global http.proxy "http://127.0.0.1:$p"
        git config --global https.proxy "http://127.0.0.1:$p"
        Write-Host "Proxy detected on port $p. Git proxy configured." -ForegroundColor Green
        $proxyFound = $true
        break
    }
}

if (-not $proxyFound) {
    git config --global --unset http.proxy 2>$null
    git config --global --unset https.proxy 2>$null
    Write-Host "No proxy detected. Git proxy disabled." -ForegroundColor Yellow
}

git add .

$commitMsg = Read-Host "Enter commit message"
git commit -m "$commitMsg"

git push origin main
Write-Host "Push completed." -ForegroundColor Cyan
