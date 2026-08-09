@echo off

call %~dp0..\system\environment.bat

git -C "%~dp0..\webui" pull 2>NUL
if %ERRORLEVEL% == 0 goto :done

git -C "%~dp0..\webui" reset --hard
git -C "%~dp0..\webui" pull

:done
pause
