// Windows port of packaging/ControlServer.swift: the local HTTP control server for AudioEngineWin, on
// 127.0.0.1:8767. Same 15 routes, same JSON shapes (see ControlServer.swift's own header comment for the exact
// contract) - capture_tool/capture_api/core.py's _native_call() needs zero changes to talk to this instead.
#pragma once

#include <atomic>
#include <thread>

#include "AudioEngineWin.h"

class ControlServerWin {
public:
    explicit ControlServerWin(unsigned short port = 8767);
    ~ControlServerWin();

    // Starts the accept loop on its own thread (mirrors ControlServer.swift's start(): accept() blocks, so it
    // cannot share a thread with anything else). Returns false if the socket could not be bound.
    bool start();

    // Releases the audio device (fading out, never a click) and stops the accept loop. Safe to call even if
    // start() was never called or already failed.
    void stop();

private:
    void run_loop();
    void handle_client(uintptr_t client_socket);
    std::string route(const std::string& method, const std::string& path, const std::string& body_json,
                       int* out_status);

    unsigned short port_;
    uintptr_t listen_socket_;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;
    AudioEngineWin engine_;
};
