# Free Port 8000 - PowerShell Script
# Usage: .\scripts\free_port.ps1 [port_number]

param(
    [int]$Port = 8000
)

Write-Host "🔍 Checking port $Port..." -ForegroundColor Cyan

# Find process using the port
$connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue

if ($connection) {
    $processId = $connection.OwningProcess
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    
    if ($process) {
        Write-Host "⚠️  Port $Port is in use by:" -ForegroundColor Yellow
        Write-Host "   Process ID: $processId" -ForegroundColor Yellow
        Write-Host "   Process Name: $($process.ProcessName)" -ForegroundColor Yellow
        Write-Host "   Path: $($process.Path)" -ForegroundColor Yellow
        
        $response = Read-Host "Do you want to kill this process? (y/N)"
        
        if ($response -eq 'y' -or $response -eq 'Y') {
            try {
                Stop-Process -Id $processId -Force
                Write-Host "✅ Process $processId killed successfully" -ForegroundColor Green
                Write-Host "✅ Port $Port is now free!" -ForegroundColor Green
            } catch {
                Write-Host "❌ Error killing process: $_" -ForegroundColor Red
            }
        } else {
            Write-Host "ℹ️  Process not killed. Port $Port is still in use." -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠️  Port $Port is in use but process not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ Port $Port is free!" -ForegroundColor Green
}
