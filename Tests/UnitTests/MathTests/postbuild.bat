@echo off
echo  +++ Linked with %2
if not X%2 == X (
  if exist %2\*.dll  xcopy/y %2\*.dll %1
)
