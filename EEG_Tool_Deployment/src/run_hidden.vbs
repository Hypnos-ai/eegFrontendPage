Set WshShell = CreateObject("WScript.Shell")
InstallPath = WshShell.ExpandEnvironmentStrings("%NEUROSYNC_PATH%")

' Construct the command
Command = "cmd.exe /k echo Debugging Command Execution && " & Chr(34) & InstallPath & "\eeg_venv\Scripts\python.exe" & Chr(34) & " " & Chr(34) & InstallPath & "\src\main.py" & Chr(34)

' Log the command for debugging
Set fso = CreateObject("Scripting.FileSystemObject")
LogFilePath = InstallPath & "\vbs_debug_log.txt"
Set logFile = fso.OpenTextFile(LogFilePath, 2, True)
logFile.WriteLine("Command to Execute: " & Command)
logFile.Close

' Execute the command
WshShell.Run Command, 0, False

' Log the end of the script execution
Set logFile = fso.OpenTextFile(LogFilePath, 8, True)
logFile.WriteLine("VBScript Executed at: " & Now)
logFile.Close

Set WshShell = Nothing
