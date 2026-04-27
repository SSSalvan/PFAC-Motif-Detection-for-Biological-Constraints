@echo off
echo ========================================================
echo   PFAC vs AC Benchmark - All Sizes
echo   Research Guide: Parallel Computing 2026
echo ========================================================
echo.

REM Ukuran N sesuai guideline dosen
set SIZES=1024 4096 16384 65536 262144 1048576 4194304 16777216

REM Buat folder hasil
if not exist results mkdir results

echo [1/3] Running PFAC benchmarks...
echo.
for %%N in (%SIZES%) do (
    echo   PFAC N=%%N ...
    PFAC.exe dummy.txt %%N > results\result_pfac_%%N.txt 2>&1
)

echo.
echo [2/3] Running AC benchmarks...
echo.
for %%N in (%SIZES%) do (
    echo   AC   N=%%N ...
    AC.exe dummy.txt %%N > results\result_ac_%%N.txt 2>&1
)

echo.
echo [3/3] Running on REAL genomic data (raw.txt)...
echo   PFAC raw.txt ...
PFAC.exe raw.txt > results\result_pfac_genomic.txt 2>&1
echo   AC   raw.txt (ini lama ~15 menit, sabar ya!)...
AC.exe raw.txt > results\result_ac_genomic.txt 2>&1

echo.
echo ========================================================
echo   SELESAI! Cek folder: results\
echo ========================================================
pause