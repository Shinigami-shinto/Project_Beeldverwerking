@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
set i=1
d:
cd D:\School\2018-2019\Project CV - Paintings\Testimgs_copy
for /f %%f in ('dir /b .\') do (
  echo renaming "%%f" to "testimg!i!.jpg"
  ren "%%f" "testimg!i!.jpg"
  set /A i=!i!+1
)
ENDLOCAL
set "i="