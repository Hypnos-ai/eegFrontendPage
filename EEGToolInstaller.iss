[Setup]
AppName=NeuroSync
AppVersion=1.0
AppPublisher=HypnosAI
DefaultDirName={autopf}\NeuroSync
DefaultGroupName=NeuroSync
OutputBaseFilename=NeuroSyncSetup
Compression=lzma2/max
SolidCompression=true
PrivilegesRequired=admin
MinVersion=6.1
OutputDir=installer_output
DisableProgramGroupPage=yes
DisableWelcomePage=yes
DisableReadyPage=yes
DisableFinishedPage=no
WizardStyle=modern

[Files]
; Python core files and DLLs (needed to create and use venv)
Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\python.exe"; DestDir: "{app}\Python"; Flags: ignoreversion
Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\pythonw.exe"; DestDir: "{app}\Python"; Flags: ignoreversion
Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\python3.dll"; DestDir: "{app}\Python"; Flags: ignoreversion
Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\python312.dll"; DestDir: "{app}\Python"; Flags: ignoreversion
Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\vcruntime140.dll"; DestDir: "{app}\Python"; Flags: ignoreversion
Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\vcruntime140_1.dll"; DestDir: "{app}\Python"; Flags: ignoreversion

; Required DLLs
Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\DLLs\*"; DestDir: "{app}\Python\DLLs"; Flags: ignoreversion recursesubdirs createallsubdirs

Source: "C:\Users\ncvn\AppData\Local\Programs\Python\Python312\Lib\*"; DestDir: "{app}\Python\Lib"; Flags: recursesubdirs createallsubdirs ignoreversion; Excludes: "site-packages\*"

Source: "eeg_tool_deployment\src\launcher.py"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "eeg_tool_deployment\src\main.py"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "eeg_tool_deployment\src\*.pyd"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs
Source: "eeg_tool_deployment\src\data\*"; DestDir: "{app}\data"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "eeg_tool_deployment\src\sample_data\*"; DestDir: "{app}\sample_data"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "eeg_tool_deployment\src\*.pkl"; DestDir: "{app}"; Flags: ignoreversion
Source: "eeg_tool_deployment\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "eeg_tool_deployment\src\NeuroSync.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "eeg_tool_deployment\src\run_hidden.vbs"; DestDir: "{app}\src"; Flags: ignoreversion



[Icons]
Name: "{group}\NeuroSync"; Filename: "{app}\eeg_venv\Scripts\python.exe"; Parameters: """{app}\src\launcher.py"""; WorkingDir: "{app}\src"; IconFilename: "{app}\NeuroSync.ico"
Name: "{userdesktop}\NeuroSync"; Filename: "{app}\eeg_venv\Scripts\python.exe"; Parameters: """{app}\src\launcher.py"""; WorkingDir: "{app}\src"; IconFilename: "{app}\NeuroSync.ico"
Name: "{group}\Uninstall NeuroSync"; Filename: "{uninstallexe}"

[Run]
; 1. Set environment variable
Filename: "cmd.exe"; Parameters: "/C setx NEUROSYNC_PATH ""{app}"" /M"; Flags: runhidden waituntilterminated

; 2. Create and setup Python environment
Filename: "{app}\Python\python.exe"; Parameters: "-m venv ""{app}\eeg_venv"""; Flags: waituntilterminated
Filename: "{app}\eeg_venv\Scripts\pip.exe"; Parameters: "install -r {app}\requirements.txt"; Flags: waituntilterminated

; 3. Generate VBS file
;Filename: "{code:GenerateVBScript}"; Flags: runhidden waituntilterminated

; 4. Launch application (optional at end of install)
;Filename: "{app}\eeg_venv\Scripts\python.exe"; Parameters: """{app}\src\launcher.py"""; Description: "Launch NeuroSync"; Flags: postinstall nowait

[UninstallDelete]
Type: filesandordirs; Name: "{app}"