// Standalone console harness for ControlServerWin: starts the native control server on 127.0.0.1:8767 and waits,
// exactly what the real WinUI3 shell will do internally (see the Windows port plan's Phase 1/6) - this lets the
// control-server <-> AudioEngineWin <-> RTEngine chain be built and exercised against the real capture_tool Python
// server (capture_api/core.py's _native_call) before the shell itself exists. Ctrl+C or closing the console stops it.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <csignal>
#include <cstdio>

#include "ControlServerWin.h"

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_signal(int) { g_stop = 1; }

LONG WINAPI crash_filter(EXCEPTION_POINTERS* info) {
    std::fprintf(stderr, "CRASH: code=0x%08lX address=%p\n",
                 info->ExceptionRecord->ExceptionCode, info->ExceptionRecord->ExceptionAddress);
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace

int main() {
    SetUnhandledExceptionFilter(crash_filter);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    for (const auto& name : AudioEngineWin::list_driver_names())
        std::printf("driver ASIO trovato: %s\n", name.c_str());
    std::fflush(stdout);

    ControlServerWin server(8767);
    if (!server.start()) {
        std::fprintf(stderr, "impossibile aprire 127.0.0.1:8767 (gia' in uso?)\n");
        return 1;
    }
    std::printf("NAM Capture native control server: 127.0.0.1:8767 (Ctrl+C per fermare)\n");
    std::fflush(stdout);
    while (!g_stop) Sleep(200);

    std::printf("arresto in corso...\n");
    server.stop();
    return 0;
}
