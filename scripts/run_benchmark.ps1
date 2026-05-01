# run_benchmark.ps1 — Jalankan berbagai ukuran N otomatis

$sizes = @(1024, 4096, 16384, 65536, 262144, 1048576, 
           4194304, 16777216, 67108864, 268435456, 
           536870912, 1073741824)

"" | Out-File result_pfac.txt
"" | Out-File result_ac.txt

foreach ($n in $sizes) {
    Write-Host ">>> N = $n"
    "=== N=$n ===" | Out-File result_pfac.txt -Append
    .\PFAC.exe raw.txt $n | Tee-Object -Append result_pfac.txt
    "=== N=$n ===" | Out-File result_ac.txt -Append
    .\AC.exe raw.txt $n | Tee-Object -Append result_ac.txt
}
Write-Host "DONE! Cek result_pfac.txt dan result_ac.txt"