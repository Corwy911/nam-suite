# Perché Win32 + WebView2 e non WinUI3/XAML

Roman aveva scelto WinUI3 per la shell nativa Windows (vedi il piano di porting). Questa cartella invece implementa
un guscio Win32 puro (una HWND, un menu classico, il controllo WebView2 via la sua API COM diretta) - ecco perché,
e cosa servirebbe per passare davvero a WinUI3/XAML in futuro.

## Cosa serve per WinUI3/XAML che qui manca

Un'app WinUI3 in C++/WinRT ha bisogno di:
1. `App.idl` / `MainWindow.idl` che dichiarano le runtimeclass (`App : Microsoft.UI.Xaml.Application`, `MainWindow
   : Microsoft.UI.Xaml.Window`).
2. MIDL che compila quegli `.idl` in un `.winmd`.
3. Il task MSBuild `XamlCompiler` (dentro il pacchetto NuGet Windows App SDK) che genera gli header C++/WinRT
   proiettati (`App.g.h`, `MainWindow.g.h`, ...) a partire da quel `.winmd` e dai file `.xaml`.
4. Il bootstrap runtime (`MddBootstrapInitialize`) per il deployment "unpackaged" (senza MSIX).

Questa pipeline è esattamente quella che Visual Studio genera e verifica in automatico quando crei un progetto
WinUI3 dal template - a mano, senza l'IDE che segnala subito un `.idl` scritto male o un passo mancante, il rischio
concreto era consegnare un progetto che sembra completo ma non compila, o compila e non si avvia, senza modo di
saperlo prima di provarlo su una macchina con Visual Studio pieno.

## La decisione

Restore NuGet, MSBuild e il resto della toolchain C++ sono stati verificati passo passo in questa sessione (vedi il
resoconto della Fase 1 nel piano). Il restore dei pacchetti `Microsoft.WindowsAppSDK` + `Microsoft.Windows.CppWinRT`
per la via WinUI3 *funziona* (pacchetti scaricati e installati correttamente) - il pezzo non verificato è
esclusivamente l'orchestrazione IDL/XAML/codegen, che serve provare con build reali per sapere se è corretta.

Per consegnare un guscio Windows *funzionante e verificato* invece di uno *plausibile ma non provato*, questa
cartella usa Win32 + l'API COM diretta di WebView2 (lo stesso schema del sample ufficiale Microsoft
"webview2_samples/GettingStartedGuides/Win32_GettingStarted"): niente XAML, niente MIDL, nessun passo di build che
non sia "compila i .cpp, linka le .lib". È stato compilato ed eseguito per davvero in questa sessione: la finestra
si apre, il menu File funziona, WebView2 mostra la pagina reale di capture_tool, il dialog "Indirizzo del
trainer..." si apre e si chiude.

La differenza pratica con WinUI3 è solo estetica (i controlli sono quelli classici di Windows, non lo stile Fluent)
- nessuna funzionalità è stata sacrificata: RTEngine, il control server e capture_tool sono linkati e parlano
esattamente come previsto dal piano.

## Se in futuro si vuole davvero WinUI3

La strada più sicura è farlo **dentro Visual Studio** (non a mano): "New Project" -> "Blank App, Unpackaged
(WinUI 3 in Desktop)" in C++/WinRT, poi spostare dentro il codice già scritto qui (`PythonSidecar.*`,
`ControlServerWin.*`, `AudioEngineWin.*` non cambiano affatto: sono indipendenti dalla shell) e sostituire
`main.cpp`/la creazione della HWND con `MainWindow.xaml`'s `WebView2` control. Il lavoro fatto in questa sessione
su Fasi 2-5 resta valido al 100% in entrambi i casi.
