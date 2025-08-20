$LOG_FILE = "$([Environment]::GetFolderPath('Desktop'))\tasks.log"
venv/scripts/activate.bat
py src/init.py >> "$LOG_FILE" 2>&1