#include "PythonSidecar.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

#include <string>

PythonSidecar::PythonSidecar(std::wstring app_root, unsigned short port) : app_root_(std::move(app_root)), port_(port) {}

PythonSidecar::~PythonSidecar() { stop(); }

namespace {
bool file_exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }
}  // namespace

bool PythonSidecar::start() {
    // Which Python: NAM_PYTHON, else the suite's shared runtime (one folder up from this app's), else the development venv.
    std::wstring python_exe = app_root_ + L"\\venv\\Scripts\\python.exe";
    wchar_t override_python[MAX_PATH]{};
    const std::wstring suite_python = app_root_ + L"\\..\\runtime\\python.exe";
    const std::wstring site = app_root_ + L"\\site";
    if (GetEnvironmentVariableW(L"NAM_PYTHON", override_python, MAX_PATH) > 0) {
        python_exe = override_python;
    } else if (file_exists(suite_python)) {
        python_exe = suite_python;
        SetEnvironmentVariableW(L"PYTHONPATH", site.c_str());      // inherited by the audio engine process too
    }
    SetEnvironmentVariableW(L"PYTHONHOME", nullptr);
    SetEnvironmentVariableW(L"PYTHONUTF8", L"1");
    SetEnvironmentVariableW(L"PYTHONDONTWRITEBYTECODE", L"1");
    std::wstring script = app_root_ + L"\\capture_tool\\app.py";
    std::wstring cmdline = L"\"" + python_exe + L"\" \"" + script + L"\" --port " + std::to_wstring(port_);

    // Log file: same per-user data folder capture_tool/app.py itself uses (see its APP_SUPPORT), so the app's own
    // log and the Python server's log sit next to each other - matches main.swift's app.log placement.
    wchar_t local_app_data[MAX_PATH]{};
    DWORD len = GetEnvironmentVariableW(L"LOCALAPPDATA", local_app_data, MAX_PATH);
    std::wstring log_dir = (len > 0) ? std::wstring(local_app_data) + L"\\NAM Capture" : app_root_;
    CreateDirectoryW(log_dir.c_str(), nullptr);
    std::wstring log_path = log_dir + L"\\capture-server.log";

    SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE log_file = CreateFileW(log_path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                                   OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = (log_file != INVALID_HANDLE_VALUE) ? log_file : nullptr;
    si.hStdError = si.hStdOutput;
    PROCESS_INFORMATION pi{};

    // CREATE_NO_WINDOW: no console flashes up, same as capture/session.py's own _NO_WINDOW for the audio engine
    // subprocess. Working directory = app_root so app.py's own relative-path assumptions (HERE = Path(__file__)...)
    // hold regardless of where the shell exe itself lives once packaged.
    BOOL ok = CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                              app_root_.c_str(), &si, &pi);
    if (log_file != INVALID_HANDLE_VALUE) CloseHandle(log_file);
    if (!ok) return false;

    // Put it in a job that is killed when the last handle closes (this program quitting - or being killed): no orphan Python,
    // no audio engine left holding the interface. Assigned while still suspended, so nothing can escape the job.
    job_handle_ = CreateJobObjectW(nullptr, nullptr);
    if (job_handle_) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(static_cast<HANDLE>(job_handle_), JobObjectExtendedLimitInformation, &info, sizeof(info));
        AssignProcessToJobObject(static_cast<HANDLE>(job_handle_), pi.hProcess);
    }
    ResumeThread(pi.hThread);
    process_handle_ = pi.hProcess;
    process_id_ = pi.dwProcessId;
    CloseHandle(pi.hThread);
    return true;
}

bool PythonSidecar::wait_until_healthy(unsigned int timeout_ms) const {
    const std::wstring path = L"/health";
    const DWORD step_ms = 200;
    for (unsigned int waited = 0; waited < timeout_ms; waited += step_ms) {
        HINTERNET session = WinHttpOpen(L"NamCaptureApp", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME,
                                         WINHTTP_NO_PROXY_BYPASS, 0);
        if (session) {
            HINTERNET connect = WinHttpConnect(session, L"127.0.0.1", port_, 0);
            if (connect) {
                HINTERNET request = WinHttpOpenRequest(connect, L"GET", path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                                        WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
                if (request) {
                    WinHttpSetTimeouts(request, 500, 500, 500, 500);
                    if (WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
                        WinHttpReceiveResponse(request, nullptr)) {
                        DWORD status = 0, status_size = sizeof(status);
                        WinHttpQueryHeaders(request, WINHTTP_QUERY_FLAG_NUMBER | WINHTTP_QUERY_STATUS_CODE,
                                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_size, WINHTTP_NO_HEADER_INDEX);
                        WinHttpCloseHandle(request);
                        WinHttpCloseHandle(connect);
                        WinHttpCloseHandle(session);
                        if (status >= 200 && status < 300) return true;
                        Sleep(step_ms);
                        continue;
                    }
                    WinHttpCloseHandle(request);
                }
                WinHttpCloseHandle(connect);
            }
            WinHttpCloseHandle(session);
        }
        Sleep(step_ms);
    }
    return false;
}

std::wstring PythonSidecar::base_url() const { return L"http://127.0.0.1:" + std::to_wstring(port_) + L"/"; }

void PythonSidecar::stop() {
    if (!process_handle_) return;
    HANDLE h = static_cast<HANDLE>(process_handle_);
    if (job_handle_) TerminateJobObject(static_cast<HANDLE>(job_handle_), 0);   // the server and its children
    // uvicorn has no clean "close the window" handle to post to; a plain TerminateProcess is what capture_tool's
    // own lifespan shutdown handler (capture_api.shutdown() releasing the audio interface) is written to tolerate -
    // same as main.swift's SIGTERM, Windows just has no signal to send an arbitrary child process.
    TerminateProcess(h, 0);
    WaitForSingleObject(h, 3000);
    CloseHandle(h);
    process_handle_ = nullptr;
    if (job_handle_) { CloseHandle(static_cast<HANDLE>(job_handle_)); job_handle_ = nullptr; }
}
