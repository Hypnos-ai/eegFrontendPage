Set WshShell = CreateObject("WScript.Shell")
' 0 => hidden window
WshShell.Run """C:\Program Files (x86)\NeuroSync\eeg_venv\Scripts\python.exe"" ""C:\Program Files (x86)\NeuroSync\src\main.py""", 0, False
