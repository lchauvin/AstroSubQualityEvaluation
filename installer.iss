; Inno Setup script for astro-eval
; Compile with: iscc installer.iss
; Requires Inno Setup 6: https://jrsoftware.org/isinfo.php
;
; What this installer does:
;   1. Installs astro-eval to %LocalAppData%\Programs\astro-eval  (no admin required)
;   2. Adds the install directory to the user PATH
;   3. Adds a right-click context menu on folders: "Analyze with astro-eval"
;   4. Creates a Start Menu shortcut
;   5. Installs example config file astro_eval.toml.example
;   6. Creates a full uninstaller

#define AppName      "astro-eval"
#define AppVersion   "0.1.0"
#define AppPublisher "Laurent"
#define AppExeName   "astro-eval.exe"
#define DistDir      "dist\astro-eval"

[Setup]
AppId={{A3F7C1B2-4D8E-4F9A-B2C3-5E6D7A8B9C0D}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=astro-eval-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayName={#AppName}
; Uncomment and set a path to add an icon:
; SetupIconFile=astro_eval\icon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Main application directory (all PyInstaller output)
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Global user config — installed to %APPDATA%\astro-eval\astro_eval.toml
; onlyifdoesntexist: never overwrite a config the user has already customised
Source: "astro_eval.toml.example"; DestDir: "{userappdata}\astro-eval"; DestName: "astro_eval.toml"; Flags: ignoreversion onlyifdoesntexist

; Keep the example in the install dir for reference
Source: "astro_eval.toml.example"; DestDir: "{app}"; Flags: ignoreversion

; Siril integration script
Source: "astro_eval_siril.py"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu shortcut — opens a terminal in the install directory
Name: "{group}\astro-eval (Command Prompt)"; Filename: "cmd.exe"; Parameters: "/k ""{app}\{#AppExeName}"" --help"; WorkingDir: "{app}"
Name: "{group}\Uninstall astro-eval"; Filename: "{uninstallexe}"

[Registry]
; ---- Right-click on a FOLDER in Explorer ----
Root: HKCU; Subkey: "Software\Classes\Directory\shell\astro-eval"; ValueType: string; ValueName: ""; ValueData: "Analyze with astro-eval"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Directory\shell\astro-eval"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKCU; Subkey: "Software\Classes\Directory\shell\astro-eval\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" ""%V"" --html --serve"

; ---- Right-click on the BACKGROUND inside a folder ----
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\astro-eval"; ValueType: string; ValueName: ""; ValueData: "Analyze with astro-eval"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\astro-eval"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#AppExeName},0"
Root: HKCU; Subkey: "Software\Classes\Directory\Background\shell\astro-eval\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" ""%V"" --html --serve"

; ---- Add install directory to user PATH ----
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Check: NeedsAddPath('{app}'); Flags: preservestringtype

[Code]
// Helper: check if the install dir is already in PATH to avoid duplicates
function NeedsAddPath(Param: string): boolean;
var
  OrigPath: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OrigPath) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Param + ';', ';' + OrigPath + ';') = 0;
end;

[Run]
; After install, offer to open a command prompt so the user can test immediately
Filename: "cmd.exe"; Parameters: "/k ""{app}\{#AppExeName}"" --help"; Description: "Open a terminal and run astro-eval --help"; Flags: postinstall skipifsilent shellexec

[UninstallRun]
; Nothing extra needed — registry entries marked with uninsdeletekey are auto-removed

[Messages]
FinishedLabel=astro-eval has been installed.%n%nRight-click any folder in Windows Explorer and choose "Analyze with astro-eval" to start.%n%nYour configuration file is at:%n  %APPDATA%\astro-eval\astro_eval.toml%n%nEdit it once to set your telescope focal length, camera pixel size, and rejection thresholds. You can also drop an astro_eval.toml in any specific session folder to override settings for that session.
