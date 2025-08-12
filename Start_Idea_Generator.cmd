
   ```bat
   @echo off
   setlocal
   cd /d "%~dp0"
   if not exist ".venv\Scripts\python.exe" (
     echo Creating virtual environment...
     py -m venv .venv || python -m venv .venv
   )
   call ".venv\Scripts\activate.bat"
   if exist "requirements.txt" (
     echo Installing requirements (first run only)...
     pip install -r requirements.txt
   )
   echo Starting Idea Generator on http://localhost:8501 ...
   start "" http://localhost:8501
   streamlit run app.py --server.port 8501
   pause
