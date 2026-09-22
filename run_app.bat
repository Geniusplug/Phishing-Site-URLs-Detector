@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
	echo Virtual environment not found. Run the setup commands in README.md first.
	exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
