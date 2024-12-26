[Setup]
AppName=NeuroSync
AppVersion=1.0
AppPublisher=HypnosAI
DefaultDirName={autopf}\NeuroSync
DefaultGroupName=NeuroSync
OutputBaseFilename=NeuroSyncSetup
SetupIconFile=eeg_tool_deployment\NeuroSync.ico
Compression=lzma2/max
SolidCompression=true
PrivilegesRequired=admin
MinVersion=10.0
OutputDir=installer_output
DisableProgramGroupPage=yes
DisableWelcomePage=yes
DisableReadyPage=yes
DisableFinishedPage=no
WizardStyle=modern

[Files]
Source: "eeg_tool_deployment\src\launcher.py"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "eeg_tool_deployment\src\main.py"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "eeg_tool_deployment\src\*.pyd"; DestDir: "{app}\src"; Flags: ignoreversion recursesubdirs
Source: "eeg_tool_deployment\src\data\*"; DestDir: "C:\NeuroSync\data"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "eeg_tool_deployment\src\sample_data\*"; DestDir: "C:\NeuroSync\sample_data"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "eeg_tool_deployment\src\*.pkl"; DestDir: "C:\NeuroSync\"; Flags: ignoreversion
Source: "eeg_tool_deployment\eeg_venv\*"; DestDir: "{app}\eeg_venv"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pyc,*.pyo,__pycache__"
Source: "eeg_tool_deployment\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "eeg_tool_deployment\src\run_hidden.vbs"; DestDir: "{app}\src"; Flags: ignoreversion

[Icons]
Name: "{group}\NeuroSync"; Filename: "{app}\eeg_venv\Scripts\python.exe"; Parameters: """{app}\src\launcher.py"""; WorkingDir: "{app}\src"; IconFilename: "{app}\NeuroSync.ico"
Name: "{userdesktop}\NeuroSync"; Filename: "{app}\eeg_venv\Scripts\python.exe"; Parameters: """{app}\src\launcher.py"""; WorkingDir: "{app}\src"; IconFilename: "{app}\NeuroSync.ico"
Name: "{group}\Uninstall NeuroSync"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\eeg_venv\Scripts\python.exe"; Parameters: """{app}\src\launcher.py"""; Description: "Launch NeuroSync"; Flags: postinstall nowait

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "C:\NeuroSync\data"
Type: filesandordirs; Name: "C:\NeuroSync\sample_data"
Type: filesandordirs; Name: "C:\NeuroSync\"