@echo off
echo Installing dependencies...
python -m pip install -r requirements.txt

echo Running the script...
python ui2.py

pause
