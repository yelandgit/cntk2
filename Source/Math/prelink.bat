@echo off

if "X%2" == "X"  (
  echo *** MKL_PATH is not defined
) else (
  xcopy/y/d %1\lib\*.lib %2
  xcopy/y/d %1\lib\*.dll %2
)
