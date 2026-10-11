; NAM Suite - one installer for NAM Trainer and NAM Capture (Inno Setup 6). Built by ..\build.py, which passes:
;   /DVersion /DStage (the staged files: runtime, trainer, capture, variants, extras) /DAssets /DOut /DHasNvidia /DHasAmd /DHasCpu
;
; Installed layout (per user, no administrator rights needed; the two apps are independent components):
;   {app}\runtime\   portable Python 3.12, shared
;   {app}\trainer\   NAM Trainer.exe (tray program), nam-web, trainers, site (packages + the PyTorch build of this PC), NAM Generati, File WAV caricati
;   {app}\capture\   NAM Capture.exe, capture_tool, site, docs
; Everything the programs create lives in {app} or in %LOCALAPPDATA%\NAM Capture: the uninstaller removes both (after asking,
; because models, WAV files and captures are the user's own work) plus the shortcuts, the registry keys and the Windows Firewall rule.

#ifndef Version
  #define Version "0.0.0"
#endif
; A TEST build (build.py iss --test-data) has its own identity - AppId, name, Start menu folder, shortcut names, registry keys - so that
; installing / uninstalling it on a PC that has the real NAM Suite can never touch the real one (its uninstall entry, shortcuts, autostart).
#ifdef CaptureData
  #define AppGuid "{{3B5D9E21-7C4A-4F6B-8A11-5E0C2D7F9B33}"
  #define AppName "NAM Suite TEST"
  #define Tag " TEST"
  #define RegSub "NAM Suite TEST"
#else
  #define AppGuid "{{6F0C2B7D-5A41-4E0B-9B7E-3D8A1C2E9F64}"
  #define AppName "NAM Suite"
  #define Tag ""
  #define RegSub "NAM Suite"
#endif
#ifndef ExtraTrainer
  #define ExtraTrainer 0
#endif
#ifndef MaxDirLen
  #define MaxDirLen 80
#endif
; Where NAM Capture keeps its data. Only test builds (build.py iss --test-data) point this somewhere else: a test uninstall must
; never touch the real folder of the person running it.
#ifdef CaptureData
  #define CapData CaptureData
#else
  #define CapData "{localappdata}\NAM Capture"
#endif

[Setup]
AppId={#AppGuid}
AppName={#AppName}
AppVersion={#Version}
AppVerName={#AppName} {#Version}
AppPublisher=Roman Asanov
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableWelcomePage=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
ShowLanguageDialog=yes
LanguageDetectionMethod=uilanguage
SetupIconFile={#Assets}\nam-suite.ico
UninstallDisplayIcon={app}\nam-suite.ico
UninstallDisplayName={#AppName}
OutputDir={#Out}
OutputBaseFilename=NAM-Suite-Setup
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=yes
DiskSliceSize=1900000000
SlicesPerDisk=1
CloseApplications=no
RestartApplications=no
UsePreviousLanguage=yes
ChangesAssociations=no

[Languages]
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
it.TypeFull=Completa: NAM Trainer e NAM Capture
it.TypeTrainer=Solo NAM Trainer (addestramento dei modelli)
it.TypeCapture=Solo NAM Capture (catturatore calibrato)
it.TypeCustom=Personalizzata
it.CompTrainer=NAM Trainer: addestra i modelli NAM
it.CompCapture=NAM Capture: cattura calibrata con la scheda audio
it.TaskDesktopTrainer=Icona di NAM Trainer sul desktop
it.TaskDesktopCapture=Icona di NAM Capture sul desktop
it.DirTooLong=Il percorso della cartella è troppo lungo: alcuni file di PyTorch hanno nomi lunghi e Windows limita i percorsi a 260 caratteri. Scegli una cartella più corta (al massimo {#MaxDirLen} caratteri), per esempio C:\NAM Suite.
it.NothingSelected=Scegli almeno un programma da installare.
it.VariantTitle=Motore di calcolo del trainer
it.VariantDesc=Quale versione di PyTorch usare per addestrare i modelli
it.VariantSub=Ho controllato il PC: %1
it.VariantNvidia=NVIDIA (CUDA): usa la scheda video NVIDIA
it.VariantAmd=AMD (ROCm): usa la scheda video AMD Radeon (Windows 11)
it.VariantCpu=Solo processore (CPU): funziona ovunque ma è molto più lento
it.VariantNoGpu=nessuna scheda NVIDIA o AMD adatta
it.RunTrainer=Avvia NAM Trainer adesso
it.RunCapture=Avvia NAM Capture adesso
it.StatusRuntime=Installo il runtime Visual C++ (Windows può chiedere il permesso di amministratore)...
it.StatusWebView=Installo il componente Microsoft Edge WebView2...
it.StatusCheck=Controllo i pacchetti e provo la scheda video (può volerci un minuto)...
it.SetupCpuFallback=La scheda AMD non è utilizzabile con PyTorch ROCm: ho installato la versione CPU (più lenta). Con un driver AMD più recente puoi reinstallare.
it.SetupNoGpu=PyTorch non vede la scheda video (driver vecchio o riavvio necessario?). NAM Trainer si avvia, ma il training non parte finché la GPU non è disponibile.
it.SetupImportFail=Uno o più pacchetti non si importano: NAM Trainer potrebbe non partire. Dettagli in:%n%1
it.UninstallAsk=NAM Suite verrà rimosso insieme a TUTTI i suoi dati, definitivamente:%n%n%1%nVuoi continuare?
it.UninstallModels=  - %1 modelli .nam (%2)
it.UninstallWavs=  - %1 file WAV del trainer (%2)
it.UninstallCapture=  - i dati del catturatore: catture, IR, modelli per «Prova dal vivo», calibrazioni (%1)
it.UninstallFirewall=Windows ha una regola del firewall per NAM Trainer (creata quando hai consentito l'accesso alla rete). Vuoi rimuoverla? Windows chiederà il permesso di amministratore.
it.UninstallClosing=Chiudo i programmi NAM...
en.TypeFull=Full: NAM Trainer and NAM Capture
en.TypeTrainer=NAM Trainer only (model training)
en.TypeCapture=NAM Capture only (calibrated capture)
en.TypeCustom=Custom
en.CompTrainer=NAM Trainer: trains NAM models
en.CompCapture=NAM Capture: calibrated capture with your audio interface
en.TaskDesktopTrainer=NAM Trainer icon on the desktop
en.TaskDesktopCapture=NAM Capture icon on the desktop
en.DirTooLong=The folder path is too long: some PyTorch files have long names and Windows limits paths to 260 characters. Choose a shorter folder (at most {#MaxDirLen} characters), for example C:\NAM Suite.
en.NothingSelected=Choose at least one program to install.
en.VariantTitle=Trainer compute engine
en.VariantDesc=Which PyTorch build to use for training models
en.VariantSub=I checked this PC: %1
en.VariantNvidia=NVIDIA (CUDA): uses the NVIDIA graphics card
en.VariantAmd=AMD (ROCm): uses the AMD Radeon graphics card (Windows 11)
en.VariantCpu=Processor only (CPU): works everywhere but is much slower
en.VariantNoGpu=no suitable NVIDIA or AMD card
en.RunTrainer=Start NAM Trainer now
en.RunCapture=Start NAM Capture now
en.StatusRuntime=Installing the Visual C++ runtime (Windows may ask for administrator permission)...
en.StatusWebView=Installing the Microsoft Edge WebView2 component...
en.StatusCheck=Checking the packages and trying the graphics card (this can take a minute)...
en.SetupCpuFallback=The AMD card cannot be used with PyTorch ROCm: the CPU version (slower) was installed. With a newer AMD driver you can reinstall.
en.SetupNoGpu=PyTorch does not see the graphics card (old driver or restart needed?). NAM Trainer starts, but training will not run until the GPU is available.
en.SetupImportFail=One or more packages do not import: NAM Trainer may not start. Details in:%n%1
en.UninstallAsk=NAM Suite will be removed together with ALL its data, permanently:%n%n%1%nDo you want to continue?
en.UninstallModels=  - %1 .nam models (%2)
en.UninstallWavs=  - %1 trainer WAV files (%2)
en.UninstallCapture=  - the capture data: captures, IRs, "live monitor" models, calibrations (%1)
en.UninstallFirewall=Windows has a firewall rule for NAM Trainer (created when you allowed network access). Remove it too? Windows will ask for administrator permission.
en.UninstallClosing=Closing the NAM programs...

[Types]
Name: "full"; Description: "{cm:TypeFull}"
Name: "trainer"; Description: "{cm:TypeTrainer}"
Name: "capture"; Description: "{cm:TypeCapture}"
Name: "custom"; Description: "{cm:TypeCustom}"; Flags: iscustom

[Components]
Name: "trainer"; Description: "{cm:CompTrainer}"; Types: full trainer custom; ExtraDiskSpaceRequired: {#ExtraTrainer}
Name: "capture"; Description: "{cm:CompCapture}"; Types: full capture custom

[Tasks]
Name: "desktoptrainer"; Description: "{cm:TaskDesktopTrainer}"; Components: trainer
Name: "desktopcapture"; Description: "{cm:TaskDesktopCapture}"; Components: capture

[Dirs]
Name: "{app}\trainer\NAM Generati"; Components: trainer
Name: "{app}\trainer\File WAV caricati"; Components: trainer

[Files]
; ---- shared Python
Source: "{#Stage}\runtime\*"; DestDir: "{app}\runtime"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: trainer capture
Source: "{#Assets}\nam-suite.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Stage}\extras\LEGGIMI.txt"; DestDir: "{app}"; Flags: ignoreversion; Languages: it
Source: "{#Stage}\extras\README-EN.txt"; DestDir: "{app}"; Flags: ignoreversion; Languages: en
; ---- trainer (the trainers folder keeps what is already installed: the page can update them)
Source: "{#Stage}\trainers-bundled\*"; DestDir: "{app}\trainer\trainers"; Flags: recursesubdirs createallsubdirs onlyifdoesntexist; Components: trainer
Source: "{#Stage}\trainer\*"; DestDir: "{app}\trainer"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: trainer
#if HasNvidia == "1"
Source: "{#Stage}\variants\nvidia\*"; DestDir: "{app}\trainer\site"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: trainer; Check: VariantIs('nvidia')
#endif
#if HasAmd == "1"
Source: "{#Stage}\variants\amd\*"; DestDir: "{app}\trainer\site"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: trainer; Check: VariantIs('amd')
#endif
#if HasCpu == "1"
Source: "{#Stage}\variants\cpu\*"; DestDir: "{app}\trainer\site"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: trainer; Check: VariantIs('cpu')
Source: "{#Stage}\variants\cpu\*"; DestDir: "{app}\trainer\_cpu_fallback"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: trainer; Check: VariantIs('amd')
#endif
; ---- capture
Source: "{#Stage}\capture\*"; DestDir: "{app}\capture"; Flags: recursesubdirs createallsubdirs ignoreversion; Components: capture
; ---- helpers (the setup folder is deleted at the end of the installation)
Source: "{#Stage}\extras\suite_setup.py"; DestDir: "{app}\setup"; Flags: ignoreversion; Components: trainer
Source: "{#Stage}\extras\vc_redist.x64.exe"; DestDir: "{app}\setup"; Flags: ignoreversion; Components: trainer capture; Check: NeedVcRedist
Source: "{#Stage}\extras\MicrosoftEdgeWebView2RuntimeInstallerX64.exe"; DestDir: "{app}\setup"; Flags: ignoreversion; Components: capture; Check: NeedWebView2
Source: "{#Stage}\extras\gpu-detect.exe"; Flags: dontcopy

[Icons]
Name: "{autodesktop}\NAM Trainer{#Tag}"; Filename: "{app}\trainer\NAM Trainer.exe"; WorkingDir: "{app}\trainer"; Tasks: desktoptrainer; Components: trainer
Name: "{autodesktop}\NAM Capture{#Tag}"; Filename: "{app}\capture\NAM Capture.exe"; WorkingDir: "{app}\capture"; Tasks: desktopcapture; Components: capture
Name: "{group}\NAM Trainer{#Tag}"; Filename: "{app}\trainer\NAM Trainer.exe"; WorkingDir: "{app}\trainer"; Components: trainer
Name: "{group}\NAM Capture{#Tag}"; Filename: "{app}\capture\NAM Capture.exe"; WorkingDir: "{app}\capture"; Components: capture
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"; IconFilename: "{app}\nam-suite.ico"

[Registry]
Root: HKCU; Subkey: "Software\{#RegSub}"; ValueType: string; ValueName: "InstallDir"; ValueData: "{app}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\{#RegSub}"; ValueType: string; ValueName: "Language"; ValueData: "{language}"
; "Start with Windows" in the tray icon's menu writes this value: removed with the program
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "NAM Trainer{#Tag}"; Flags: uninsdeletevalue

[Run]
Filename: "{app}\trainer\NAM Trainer.exe"; Description: "{cm:RunTrainer}"; WorkingDir: "{app}\trainer"; Flags: nowait postinstall skipifsilent; Components: trainer
Filename: "{app}\capture\NAM Capture.exe"; Description: "{cm:RunCapture}"; WorkingDir: "{app}\capture"; Flags: nowait postinstall skipifsilent unchecked; Components: capture

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{#CapData}"
Type: filesandordirs; Name: "{localappdata}\NAM Suite"
Type: filesandordirs; Name: "{userappdata}\NAM Suite"

[Code]
var
  VariantPage: TInputOptionWizardPage;
  VariantNames: array of String;
  DetectedVariant, DetectedGpu, DetectedCpu, DetectedReason: String;
  ChosenVariant: String;

// ----------------------------------------------------------------------------------------- helpers
function ValueOf(const Lines: TArrayOfString; const Key: String): String;
var
  I, P: Integer;
begin
  Result := '';
  for I := 0 to GetArrayLength(Lines) - 1 do
  begin
    P := Pos('=', Lines[I]);
    if (P > 0) and (Copy(Lines[I], 1, P - 1) = Key) then
    begin
      Result := Trim(Copy(Lines[I], P + 1, MaxInt));
      Exit;
    end;
  end;
end;

function FormatSize(Bytes: Int64): String;
begin
  if Bytes >= 1073741824 then Result := Format('%.1f GB', [Bytes / 1073741824.0])
  else Result := Format('%d MB', [(Bytes + 524288) div 1048576]);
end;

// Files (with the given extension, or all if Ext = '') under Dir, recursively.
procedure ScanDir(const Dir, Ext: String; var Count: Integer; var Size: Int64);
var
  R: TFindRec;
begin
  if FindFirst(Dir + '\*', R) then
  begin
    try
      repeat
        if (R.Name <> '.') and (R.Name <> '..') then
        begin
          if (R.Attributes and FILE_ATTRIBUTE_DIRECTORY) <> 0 then
            ScanDir(Dir + '\' + R.Name, Ext, Count, Size)
          else if (Ext = '') or (CompareText(ExtractFileExt(R.Name), Ext) = 0) then
          begin
            Count := Count + 1;
            Size := Size + (Int64(R.SizeHigh) shl 32) + R.SizeLow;
          end;
        end;
      until not FindNext(R);
    finally
      FindClose(R);
    end;
  end;
end;

procedure StopApps(const AppDir: String);
var
  Code: Integer;
  Trainer: String;
begin
  Trainer := AppDir + '\trainer\NAM Trainer.exe';
  if FileExists(Trainer) then
    Exec(Trainer, '--quit', '', SW_HIDE, ewWaitUntilTerminated, Code);
  // only the NAM Capture of THIS installation (another copy, e.g. a test build next to the real one, must not be touched)
  Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
       '-NoProfile -NonInteractive -Command "Get-Process -Name ''NAM Capture'' -ErrorAction SilentlyContinue | Where-Object { $_.Path -like ''' + AppDir + '\*'' } | Stop-Process -Force"',
       '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

// ----------------------------------------------------------------------------------------- what is already on the PC
function NeedVcRedist: Boolean;
var
  Installed, Major, Minor: Cardinal;
begin
  Result := True;
  if RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1) then
    if RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Major', Major) and
       RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Minor', Minor) then
      Result := not ((Major > 14) or ((Major = 14) and (Minor >= 38)));
end;

function WebView2Version(Root: Integer): String;
begin
  Result := '';
  if not RegQueryStringValue(Root, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Result) then
    if not RegQueryStringValue(Root, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Result) then
      Result := '';
end;

function NeedWebView2: Boolean;
var
  V: String;
begin
  V := WebView2Version(HKLM);
  if (V = '') or (V = '0.0.0.0') then V := WebView2Version(HKCU);
  Result := (V = '') or (V = '0.0.0.0');
end;

function VariantIs(const Name: String): Boolean;
begin
  Result := ChosenVariant = Name;
end;

// ----------------------------------------------------------------------------------------- wizard
procedure DetectHardware;
var
  Tmp: String;
  Lines: TArrayOfString;
  Code: Integer;
begin
  DetectedVariant := 'cpu';
  ExtractTemporaryFile('gpu-detect.exe');
  Tmp := ExpandConstant('{tmp}\hw.txt');
  if Exec(ExpandConstant('{tmp}\gpu-detect.exe'), ActiveLanguage + ' "' + Tmp + '"', '', SW_HIDE, ewWaitUntilTerminated, Code) then
    if LoadStringsFromFile(Tmp, Lines) then
    begin
      DetectedVariant := ValueOf(Lines, 'variant');
      DetectedGpu := ValueOf(Lines, 'gpu');
      DetectedCpu := ValueOf(Lines, 'cpu');
      DetectedReason := ValueOf(Lines, 'reason');
    end;
  if DetectedVariant = '' then DetectedVariant := 'cpu';
end;

procedure AddVariant(const Name, Caption: String; var Default: Integer);
var
  N: Integer;
begin
  N := GetArrayLength(VariantNames);
  SetArrayLength(VariantNames, N + 1);
  VariantNames[N] := Name;
  VariantPage.Add(Caption);
  if Name = DetectedVariant then Default := N;
end;

procedure InitializeWizard;
var
  Default, N: Integer;
  Info: String;
begin
  DetectHardware;
  Info := DetectedReason;
  if DetectedGpu <> '' then Info := DetectedGpu + ' - ' + Info;
  VariantPage := CreateInputOptionPage(wpSelectComponents, CustomMessage('VariantTitle'), CustomMessage('VariantDesc'),
    FmtMessage(CustomMessage('VariantSub'), [Info]), True, False);
  Default := -1;
#if HasNvidia == "1"
  AddVariant('nvidia', CustomMessage('VariantNvidia'), Default);
#endif
#if HasAmd == "1"
  AddVariant('amd', CustomMessage('VariantAmd'), Default);
#endif
#if HasCpu == "1"
  AddVariant('cpu', CustomMessage('VariantCpu'), Default);
#endif
  if Default < 0 then Default := GetArrayLength(VariantNames) - 1;     // the detected build is not in this package: the last (CPU) one
  // unattended / test installs: /VARIANT=nvidia|amd|cpu
  for N := 0 to GetArrayLength(VariantNames) - 1 do
    if VariantNames[N] = ExpandConstant('{param:VARIANT|}') then Default := N;
  VariantPage.SelectedValueIndex := Default;
  ChosenVariant := VariantNames[Default];
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  if (VariantPage <> nil) and (PageID = VariantPage.ID) then
    Result := (not WizardIsComponentSelected('trainer')) or (GetArrayLength(VariantNames) < 2);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectComponents then
  begin
    if not (WizardIsComponentSelected('trainer') or WizardIsComponentSelected('capture')) then
    begin
      MsgBox(CustomMessage('NothingSelected'), mbError, MB_OK);
      Result := False;
    end;
  end;
  if CurPageID = wpSelectDir then
    if Length(WizardDirValue) > {#MaxDirLen} then
    begin
      MsgBox(CustomMessage('DirTooLong'), mbError, MB_OK);
      Result := False;
    end;
  if (VariantPage <> nil) and (CurPageID = VariantPage.ID) then
    ChosenVariant := VariantNames[VariantPage.SelectedValueIndex];
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  App: String;
begin
  Result := '';
  if GetArrayLength(VariantNames) = 1 then ChosenVariant := VariantNames[0];
  App := ExpandConstant('{app}');
  StopApps(App);
  // an update starts from clean program folders: old packages of another PyTorch build must not stay mixed in
  if WizardIsComponentSelected('trainer') then
  begin
    DelTree(App + '\trainer\site', True, True, True);
    DelTree(App + '\trainer\_cpu_fallback', True, True, True);
  end;
  DelTree(App + '\runtime', True, True, True);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  App, Py, Params, Log: String;
  Code: Integer;
begin
  if CurStep <> ssPostInstall then Exit;
  App := ExpandConstant('{app}');

  if FileExists(App + '\setup\vc_redist.x64.exe') then
  begin
    WizardForm.StatusLabel.Caption := CustomMessage('StatusRuntime');
    ShellExec('runas', App + '\setup\vc_redist.x64.exe', '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, Code);
  end;
  if FileExists(App + '\setup\MicrosoftEdgeWebView2RuntimeInstallerX64.exe') then
  begin
    WizardForm.StatusLabel.Caption := CustomMessage('StatusWebView');
    if not Exec(App + '\setup\MicrosoftEdgeWebView2RuntimeInstallerX64.exe', '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, Code) or (Code <> 0) then
      ShellExec('runas', App + '\setup\MicrosoftEdgeWebView2RuntimeInstallerX64.exe', '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, Code);
  end;

  if WizardIsComponentSelected('trainer') then
  begin
    WizardForm.StatusLabel.Caption := CustomMessage('StatusCheck');
    Py := App + '\runtime\python.exe';
    Params := '"' + App + '\setup\suite_setup.py" --trainer "' + App + '\trainer" --variant ' + ChosenVariant +
              ' --gpu "' + DetectedGpu + '" --cpu "' + DetectedCpu + '" --lang ' + ActiveLanguage;
    Log := App + '\trainer\nam-web\installazione.log';
    if Exec(Py, Params, App + '\trainer', SW_HIDE, ewWaitUntilTerminated, Code) then
    begin
      if not WizardSilent then
      begin
        if Code = 10 then MsgBox(CustomMessage('SetupCpuFallback'), mbInformation, MB_OK)
        else if Code = 11 then MsgBox(CustomMessage('SetupNoGpu'), mbInformation, MB_OK)
        else if Code >= 20 then MsgBox(FmtMessage(CustomMessage('SetupImportFail'), [Log]), mbError, MB_OK);
      end;
    end;
  end;
  // both programs on this PC: NAM Capture already knows where the trainer is ("Invia al NAM Trainer" works at once)
  if WizardIsComponentSelected('trainer') and WizardIsComponentSelected('capture') and
     not FileExists(App + '\capture\capture_tool\settings.json') then
    SaveStringToFile(App + '\capture\capture_tool\settings.json', '{"lan": true, "trainer_host": "localhost"}', False);
  DelTree(App + '\setup', True, True, True);
end;

// ----------------------------------------------------------------------------------------- uninstall
function FirewallRuleCount(const App: String): Integer;
var
  Tmp, Cmd: String;
  Code: Integer;
  S: AnsiString;
begin
  Result := 0;
  Tmp := ExpandConstant('{tmp}\fw.txt');
  Cmd := '-NoProfile -NonInteractive -Command "try { (Get-NetFirewallApplicationFilter -ErrorAction Stop | Where-Object { $_.Program -like ''' +
         App + '\*'' } | Measure-Object).Count | Out-File -Encoding ascii -FilePath ''' + Tmp + ''' } catch { 0 | Out-File -Encoding ascii -FilePath ''' + Tmp + ''' }"';
  if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Cmd, '', SW_HIDE, ewWaitUntilTerminated, Code) then
    if LoadStringFromFile(Tmp, S) then
      Result := StrToIntDef(Trim(String(S)), 0);
end;

procedure RemoveFirewallRules(const App: String);
var
  Cmd: String;
  Code: Integer;
begin
  Cmd := '-NoProfile -Command "Get-NetFirewallApplicationFilter | Where-Object { $_.Program -like ''' + App +
         '\*'' } | Get-NetFirewallRule | Remove-NetFirewallRule"';
  ShellExec('runas', ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Cmd, '', SW_HIDE, ewWaitUntilTerminated, Code);
end;

function InitializeUninstall: Boolean;
var
  App, Text: String;
  ModelCount, WavCount, CapCount: Integer;
  ModelSize, WavSize, CapSize: Int64;
begin
  Result := True;
  App := ExpandConstant('{app}');
  ModelCount := 0; WavCount := 0; CapCount := 0; ModelSize := 0; WavSize := 0; CapSize := 0;
  ScanDir(App + '\trainer\NAM Generati', '.nam', ModelCount, ModelSize);
  ScanDir(App + '\trainer\File WAV caricati', '.wav', WavCount, WavSize);
  ScanDir(ExpandConstant('{#CapData}'), '', CapCount, CapSize);
  Text := '';
  if ModelCount > 0 then Text := Text + FmtMessage(CustomMessage('UninstallModels'), [IntToStr(ModelCount), FormatSize(ModelSize)]) + #13#10;
  if WavCount > 0 then Text := Text + FmtMessage(CustomMessage('UninstallWavs'), [IntToStr(WavCount), FormatSize(WavSize)]) + #13#10;
  if CapCount > 0 then Text := Text + FmtMessage(CustomMessage('UninstallCapture'), [FormatSize(CapSize)]) + #13#10;
  if (Text <> '') and not UninstallSilent then
    Result := SuppressibleMsgBox(FmtMessage(CustomMessage('UninstallAsk'), [Text]), mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDYES) = IDYES;
  if Result then
  begin
    StopApps(App);
    if (not UninstallSilent) and (FirewallRuleCount(App) > 0) then
      if MsgBox(CustomMessage('UninstallFirewall'), mbConfirmation, MB_YESNO) = IDYES then
        RemoveFirewallRules(App);
  end;
end;
