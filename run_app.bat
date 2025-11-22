@echo off
echo ======================================================
echo        AI STOCK APP LAUNCHER (ATTEMPTING FIX)
echo ======================================================
echo.

REM 1. Move to the current folder (necessary for stock_app.py to be found)
cd /d "%~dp0"
echo Current directory set to: %CD%
echo.

REM 2. Install/Update libraries (Forcefully showing output this time)
echo --- STARTING INSTALLATION (This may take a minute) ---
echo.
echo Executing: "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" -m pip install numpy pandas yfinance matplotlib streamlit scikit-learn tensorflow
echo.

REM The 'call' command allows the script to continue after this command finishes.
REM The '||' ensures the echo command runs if the installation FAILED.
"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" -m pip install numpy pandas yfinance matplotlib streamlit scikit-learn tensorflow || (echo !!! INSTALLATION FAILED. SEE ERROR ABOVE. !!!)

echo.
echo --- INSTALLATION FINISHED/SKIPPED ---
echo.

REM 3. Run the App
echo --- ATTEMPTING TO LAUNCH STREAMLIT APP ---
echo Executing: "C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run stock_app.py
echo.

"C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run stock_app.py

echo.
echo !!! STREAMLIT SERVER CRASHED OR FAILED TO START !!!
echo If you see this, something went wrong with the Python script or Streamlit itself.
echo !!! PLEASE COPY ALL THE TEXT ABOVE THIS LINE AND SHARE IT !!!

REM 4. Pause so you can see errors if it crashes
pause
