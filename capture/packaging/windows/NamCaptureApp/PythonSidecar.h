// Launches and supervises the capture_tool Python server as a child process - Windows counterpart of what
// packaging/main.swift's AppDelegate does inline (spawn "venv/bin/python app.py --port 8766", poll /health, kill
// on quit). Kept as its own small class so main.cpp stays focused on the window/WebView2 plumbing.
#pragma once

#include <string>

class PythonSidecar {
public:
    // app_root: the folder containing capture_tool/ and venv/ (the project root in development; the app's own
    // install folder once packaged - see the Windows port plan's Phase 7).
    explicit PythonSidecar(std::wstring app_root, unsigned short port = 8766);
    ~PythonSidecar();
    PythonSidecar(const PythonSidecar&) = delete;
    PythonSidecar& operator=(const PythonSidecar&) = delete;

    // Python used: $NAM_PYTHON, else <app_root>\..\runtime\python.exe (NAM Suite layout; packages in <app_root>\site via
    // PYTHONPATH), else the development venv. Starts <python> capture_tool\app.py --port <port>, output redirected to app.log in the
    // per-user data folder (see capture_tool/app.py's own APP_SUPPORT). Returns false if CreateProcess itself
    // failed (not if the server is merely slow to answer - see wait_until_healthy for that).
    bool start();

    // Polls GET http://127.0.0.1:<port>/health until it answers 2xx or timeout_ms elapses. Same reasoning as
    // main.swift polling before loading the page into the WKWebView: the window must not navigate to the URL
    // before uvicorn is actually listening.
    bool wait_until_healthy(unsigned int timeout_ms = 15000) const;

    std::wstring base_url() const;

    // SIGTERM-equivalent (Windows has no real SIGTERM for arbitrary processes: this closes the process's main
    // window if any, then falls back to TerminateProcess after a short grace period) - called on app shutdown.
    void stop();

private:
    std::wstring app_root_;
    unsigned short port_;
    void* process_handle_ = nullptr;   // HANDLE, kept void* so windows.h stays out of this header
    void* job_handle_ = nullptr;       // job object (kill-on-close): the server AND what it spawns (the audio engine) die with us
    unsigned long process_id_ = 0;
};
