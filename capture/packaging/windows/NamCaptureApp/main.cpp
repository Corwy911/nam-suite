// NAM Capture - Windows shell: a plain Win32 window hosting WebView2 (no browser chrome), the capture_tool Python
// subprocess (PythonSidecar.h) and the native control server (ControlServerWin.h) - the Windows counterpart of
// packaging/main.swift's AppDelegate. See NamCaptureApp.vcxproj's header comment for why this is plain Win32 +
// WebView2 rather than WinUI3/XAML.
//
// WebView2 already shows native confirm()/alert()/prompt() dialogs and a native file-open dialog for
// <input type=file> with NO host code at all (unlike WKWebView on Mac, which needs the WKUIDelegate methods this
// file therefore does NOT need to implement) - verified against the real page, see the Windows port plan's Phase 1.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <wrl.h>

#include <WebView2.h>

#include <string>

#include "ControlServerWin.h"
#include "PythonSidecar.h"

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;

namespace {

constexpr wchar_t kWindowClass[] = L"NamCaptureAppWindow";
constexpr wchar_t kWindowTitle[] = L"NAM Capture";
constexpr UINT kIdTrainerAddress = 1001;
constexpr UINT kIdBrowseCaptures = 1002;
constexpr UINT kIdExit = 1003;
constexpr UINT kIdTrainerBrowse = 1004;
constexpr UINT kIdEditTrainerHost = 2001;

std::wstring g_app_root;
PythonSidecar* g_sidecar = nullptr;
ControlServerWin* g_control_server = nullptr;
ComPtr<ICoreWebView2Controller> g_webview_controller;
ComPtr<ICoreWebView2> g_webview;
std::wstring g_trainer_host = L"";   // empty = not configured yet, mirrors settings.json's own default

std::wstring exe_dir() {
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring s(path);
    size_t pos = s.find_last_of(L"\\/");
    return (pos == std::wstring::npos) ? s : s.substr(0, pos);
}

// The packaged layout (packaging\windows\build_app.ps1's output) has capture_tool\/venv\ right next to the exe;
// the development layout (built straight from NamCaptureApp.vcxproj, this project's own x64\Release\) has them
// five levels up at the repo root instead. Detected by what is actually on disk, not by a build-config flag, so
// the same exe works if copied around either way.
std::wstring find_app_root(const std::wstring& start) {
    if (GetFileAttributesW((start + L"\\capture_tool").c_str()) != INVALID_FILE_ATTRIBUTES) return start;
    std::wstring dev_root = start + L"\\..\\..\\..\\..\\..";
    if (GetFileAttributesW((dev_root + L"\\capture_tool").c_str()) != INVALID_FILE_ATTRIBUTES) return dev_root;
    return start;   // neither found: start() below will fail with a clear error instead of silently misbehaving
}

// Click (not Enter) on a BUTTON child delivers its WM_COMMAND/BN_CLICKED to the parent via SendMessage, which
// calls the parent's window procedure directly and synchronously - it never goes through the message queue, so a
// bare GetMessageW loop peeking at queued messages (what this dialog used to do, with wc.lpfnWndProc =
// DefWindowProcW) can never see it. Only Enter (BS_DEFPUSHBUTTON, via IsDialogMessageW's own PostMessage) showed
// up that way, which is why clicking OK with the mouse looked like it "did nothing" while testing by keyboard did
// not catch it. The fix is a real window procedure for the dialog that reacts to WM_COMMAND itself.
struct TrainerHostDlgState {
    HWND edit = nullptr;
    bool done = false;
    bool accepted = false;
};

LRESULT CALLBACK TrainerHostDlgProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    auto* state = reinterpret_cast<TrainerHostDlgState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (msg == WM_COMMAND && state && (LOWORD(wparam) == IDOK || LOWORD(wparam) == IDCANCEL)) {
        state->accepted = LOWORD(wparam) == IDOK;
        state->done = true;
        return 0;
    }
    if (msg == WM_CLOSE && state) {
        state->accepted = false;
        state->done = true;
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wparam, lparam);
}

// A minimal modal "type some text" dialog (Win32 has no built-in input box) - the Windows equivalent of
// main.swift's NSAlert + NSTextField for "Indirizzo del trainer...". A plain modal HWND built from
// CreateWindowExW child controls, with its own tiny message loop - no resource script (.rc) or DLGTEMPLATE needed.
//
// The app is manifested PerMonitorV2 DPI-aware (app.manifest, needed for WebView2), which means Windows does NOT
// rescale this window for us the way a non-DPI-aware app would get for free - every pixel value below is at 96 DPI
// and MUST be scaled by hand, or the whole dialog (and its text) shrinks to a cramped, clipped mess on any display
// above 100% scaling: exactly what made the label wrap/clip and pushed the buttons to the edge (Roman's screenshot).
bool show_trainer_host_dialog(HWND owner, std::wstring* out_value) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = TrainerHostDlgProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"NamCaptureTrainerHostDialog";
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    RegisterClassW(&wc);   // harmless if already registered (ERROR_CLASS_ALREADY_EXISTS is ignored)

    UINT dpi = GetDpiForWindow(owner);
    auto S = [dpi](int v) { return MulDiv(v, dpi, 96); };
    HFONT font = CreateFontW(-MulDiv(9, dpi, 72), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS,
                              L"Segoe UI");

    RECT client{0, 0, S(320), S(150)};
    DWORD style = WS_POPUP | WS_CAPTION | WS_SYSMENU;
    AdjustWindowRectEx(&client, style, FALSE, WS_EX_DLGMODALFRAME);
    HWND dlg = CreateWindowExW(WS_EX_DLGMODALFRAME, wc.lpszClassName, L"Indirizzo del trainer", style,
                                CW_USEDEFAULT, CW_USEDEFAULT, client.right - client.left, client.bottom - client.top,
                                owner, nullptr, wc.hInstance, nullptr);
    HWND label = CreateWindowExW(0, L"STATIC", L"Indirizzo del NAM Trainer (solo l'IP, es. 192.168.1.42; se \u00e8 questo PC: localhost):",
                                  WS_CHILD | WS_VISIBLE, S(12), S(12), S(300), S(34), dlg, nullptr, wc.hInstance, nullptr);
    HWND edit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", g_trainer_host.c_str(), WS_CHILD | WS_VISIBLE | WS_BORDER,
                                 S(12), S(50), S(296), S(24), dlg, reinterpret_cast<HMENU>(static_cast<UINT_PTR>(kIdEditTrainerHost)),
                                 wc.hInstance, nullptr);
    HWND ok = CreateWindowExW(0, L"BUTTON", L"OK", WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON,
                               S(128), S(90), S(84), S(28), dlg, reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDOK)), wc.hInstance, nullptr);
    HWND cancel = CreateWindowExW(0, L"BUTTON", L"Annulla", WS_CHILD | WS_VISIBLE | WS_TABSTOP,
                                   S(220), S(90), S(84), S(28), dlg, reinterpret_cast<HMENU>(static_cast<UINT_PTR>(IDCANCEL)), wc.hInstance, nullptr);
    for (HWND ctrl : {label, edit, ok, cancel}) SendMessageW(ctrl, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);

    TrainerHostDlgState state;
    state.edit = edit;
    SetWindowLongPtrW(dlg, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&state));

    SetFocus(edit);
    ShowWindow(dlg, SW_SHOW);
    EnableWindow(owner, FALSE);

    MSG msg;
    while (!state.done && GetMessageW(&msg, nullptr, 0, 0)) {
        if (!IsDialogMessageW(dlg, &msg)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    }
    if (state.accepted) {
        wchar_t buf[256]{};
        GetWindowTextW(edit, buf, 256);
        *out_value = buf;
    }
    EnableWindow(owner, TRUE);
    DestroyWindow(dlg);
    DeleteObject(font);
    SetForegroundWindow(owner);
    return state.accepted;
}

// "Sfoglia catture...": opens File WAV caricati/Catture in Explorer, same as main.swift's NSOpenPanel-based
// action - here it is simpler (no picker, just reveal), matching capture_api/takes.py's own reveal-in-folder route
// which already branches to `explorer` on win32.
void browse_captures_folder() {
    wchar_t local_app_data[MAX_PATH]{};
    GetEnvironmentVariableW(L"LOCALAPPDATA", local_app_data, MAX_PATH);
    std::wstring path = std::wstring(local_app_data) + L"\\NAM Capture\\File WAV caricati\\Catture";
    SHCreateDirectoryExW(nullptr, path.c_str(), nullptr);
    ShellExecuteW(nullptr, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

// A single-quoted JS string literal is built by hand below (ExecuteScript takes a script, not arguments), so
// anything that would break out of it - a backslash, a quote, a line break - must be escaped first. Host names and
// filesystem paths from a real dialog never contain these in practice, but a hand-typed trainer address could.
std::wstring js_string_literal(const std::wstring& s) {
    std::wstring out = L"'";
    for (wchar_t c : s) {
        if (c == L'\\' || c == L'\'') out += L'\\';
        if (c == L'\r' || c == L'\n') continue;
        out += c;
    }
    out += L'\'';
    return out;
}

void run_script(const std::wstring& script) {
    if (g_webview) g_webview->ExecuteScript(script.c_str(), nullptr);
}

// "Invia catture al trainer...": a folder picker (any capture folder made before this run, or a "Catture"-like
// folder holding several) for capture_api.py's /trainer/scan - the Windows equivalent of main.swift's
// NSOpenPanel(canChooseDirectories) call feeding window.trainerBrowseFolder on the page.
bool pick_folder(HWND owner, std::wstring* out_path) {
    Microsoft::WRL::ComPtr<IFileOpenDialog> dialog;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return false;
    DWORD opts = 0;
    dialog->GetOptions(&opts);
    dialog->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    if (FAILED(dialog->Show(owner))) return false;   // includes the user cancelling - not an error
    Microsoft::WRL::ComPtr<IShellItem> item;
    if (FAILED(dialog->GetResult(&item))) return false;
    PWSTR path = nullptr;
    if (FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) return false;
    *out_path = path;
    CoTaskMemFree(path);
    return true;
}

void resize_webview(HWND hwnd) {
    if (!g_webview_controller) return;
    RECT bounds;
    GetClientRect(hwnd, &bounds);
    g_webview_controller->put_Bounds(bounds);
}

void init_webview(HWND hwnd) {
    // The page's own profile (cache, local storage) lives in the per-user data folder, not next to the exe: the install folder
    // may not be writable, and the uninstaller removes this folder together with the rest of the app's data.
    wchar_t local_app_data[MAX_PATH]{};
    GetEnvironmentVariableW(L"LOCALAPPDATA", local_app_data, MAX_PATH);
    const std::wstring profile = std::wstring(local_app_data) + L"\\NAM Capture\\WebView2";
    SHCreateDirectoryExW(nullptr, profile.c_str(), nullptr);
    HRESULT hr = CreateCoreWebView2EnvironmentWithOptions(
        nullptr, profile.c_str(), nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [hwnd](HRESULT, ICoreWebView2Environment* env) -> HRESULT {
                env->CreateCoreWebView2Controller(
                    hwnd, Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                              [hwnd](HRESULT, ICoreWebView2Controller* controller) -> HRESULT {
                                  if (controller) {
                                      g_webview_controller = controller;
                                      g_webview_controller->get_CoreWebView2(&g_webview);
                                      resize_webview(hwnd);
                                      if (g_webview && g_sidecar)
                                          g_webview->Navigate(g_sidecar->base_url().c_str());
                                  }
                                  return S_OK;
                              })
                              .Get());
                return S_OK;
            })
            .Get());
    if (FAILED(hr))
        MessageBoxW(hwnd, L"Manca Microsoft Edge WebView2 Runtime, che serve a mostrare la finestra.\nInstallalo da https://go.microsoft.com/fwlink/p/?LinkId=2124703 e riapri NAM Capture.",
                    kWindowTitle, MB_ICONERROR);
}

void build_menu(HWND hwnd) {
    HMENU menu_bar = CreateMenu();
    HMENU file_menu = CreatePopupMenu();
    AppendMenuW(file_menu, MF_STRING, kIdTrainerAddress, L"Indirizzo del trainer...");
    AppendMenuW(file_menu, MF_STRING, kIdBrowseCaptures, L"Sfoglia catture...");
    AppendMenuW(file_menu, MF_STRING, kIdTrainerBrowse, L"Invia catture al trainer...");
    AppendMenuW(file_menu, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(file_menu, MF_STRING, kIdExit, L"Esci");
    AppendMenuW(menu_bar, MF_POPUP, reinterpret_cast<UINT_PTR>(file_menu), L"File");
    SetMenu(hwnd, menu_bar);
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
        case WM_SIZE:
            resize_webview(hwnd);
            return 0;
        case WM_COMMAND:
            switch (LOWORD(wparam)) {
                case kIdTrainerAddress: {
                    std::wstring value;
                    if (show_trainer_host_dialog(hwnd, &value)) {
                        g_trainer_host = value;
                        run_script(L"window.trainerSetHost && window.trainerSetHost(" + js_string_literal(value) + L")");
                    }
                    return 0;
                }
                case kIdBrowseCaptures:
                    browse_captures_folder();
                    return 0;
                case kIdTrainerBrowse: {
                    std::wstring path;
                    if (pick_folder(hwnd, &path)) {
                        run_script(L"window.trainerBrowseFolder && window.trainerBrowseFolder(" + js_string_literal(path) + L")");
                    }
                    return 0;
                }
                case kIdExit:
                    DestroyWindow(hwnd);
                    return 0;
            }
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
}

}  // namespace

int APIENTRY wWinMain(HINSTANCE instance, HINSTANCE, LPWSTR, int show_cmd) {
    // One copy at a time (ports 8766/8767 and the audio interface cannot be shared): a second launch brings the first forward.
    HANDLE single = CreateMutexW(nullptr, TRUE, L"Local\\NAMSuite.Capture");
    if (single && GetLastError() == ERROR_ALREADY_EXISTS) {
        if (HWND other = FindWindowW(kWindowClass, nullptr)) {
            if (IsIconic(other)) ShowWindow(other, SW_RESTORE);
            SetForegroundWindow(other);
        }
        return 0;
    }
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

    g_app_root = find_app_root(exe_dir());

    PythonSidecar sidecar(g_app_root, 8766);
    g_sidecar = &sidecar;
    if (!sidecar.start()) {
        MessageBoxW(nullptr, L"Impossibile avviare il server Python (capture_tool/app.py).", kWindowTitle, MB_ICONERROR);
        return 1;
    }

    ControlServerWin control_server(8767);
    g_control_server = &control_server;
    control_server.start();   // non-fatal if it fails to bind (e.g. port already in use) - /open just answers with an error, same as Mac

    if (!sidecar.wait_until_healthy()) {
        MessageBoxW(nullptr, L"Il server Python non risponde su /health.", kWindowTitle, MB_ICONERROR);
        return 1;
    }

    WNDCLASSW wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = instance;
    wc.lpszClassName = kWindowClass;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(1));
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(0, kWindowClass, kWindowTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                                 1100, 900, nullptr, nullptr, instance, nullptr);
    build_menu(hwnd);
    init_webview(hwnd);
    ShowWindow(hwnd, show_cmd);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    control_server.stop();
    sidecar.stop();
    CoUninitialize();
    return static_cast<int>(msg.wParam);
}
