@echo off
title MNIST Digit Recognition - Softmax Regression
cls

:MENU
echo ============================================================
echo MNIST Digit Recognition - Softmax Regression
echo ============================================================
echo 1. Install Dependencies
echo 2. Train Model
echo 3. Run Web Application
echo 4. Run All (Install -^> Train -^> App)
echo 5. Exit
echo ============================================================
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto INSTALL
if "%choice%"=="2" goto TRAIN
if "%choice%"=="3" goto RUN
if "%choice%"=="4" goto ALL
if "%choice%"=="5" goto EXIT

echo Invalid choice. Please try again.
pause
cls
goto MENU

:INSTALL
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
pause
cls
goto MENU

:TRAIN
echo.
echo Training model...
python src/train.py
echo.
pause
cls
goto MENU

:RUN
echo.
echo Starting Web Application...
python app.py
goto MENU

:ALL
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Training model...
python src/train.py
echo.
echo Starting Web Application...
python app.py
goto MENU

:EXIT
exit
