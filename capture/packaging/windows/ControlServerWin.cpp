// See ControlServerWin.h. Built and verified end to end, including a real ASIO open/state/stop cycle against a
// Focusrite Scarlett 18i20 (see packaging/windows/build_controlserver_asio.bat and AudioEngineWin.cpp's own build
// status comment for what that shook out); mirrors ControlServer.swift's accept-loop/parsing/routing shape as
// closely as C++/Winsock allows, so a review against that file doubles as a review of this one.
#include "ControlServerWin.h"

#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <objbase.h>
#ifdef _MSC_VER
#pragma comment(lib, "Ws2_32.lib")
#endif

#include "../../native/third_party/nam/nlohmann/json.hpp"

#include <cstring>
#include <iostream>
#include <sstream>

using nlohmann::json;

namespace {

std::vector<int> json_int_array(const json& j, const char* key) {
    std::vector<int> out;
    if (j.contains(key) && j[key].is_array())
        for (const auto& v : j[key]) if (v.is_number_integer()) out.push_back(v.get<int>());
    return out;
}

// {"legs": ["A","B"]} -> bit 0 = B (the model's chain, the default and the only one outside player mode), bit 1 = A (the real gear's)
int legs_mask(const json& j) {
    if (!j.contains("legs") || !j["legs"].is_array()) return 1;
    int mask = 0;
    for (const auto& v : j["legs"]) {
        if (v.is_string() && v.get<std::string>() == "B") mask |= 1;
        if (v.is_string() && v.get<std::string>() == "A") mask |= 2;
    }
    return mask;
}

}  // namespace

ControlServerWin::ControlServerWin(unsigned short port) : port_(port), listen_socket_(0) {}

ControlServerWin::~ControlServerWin() { stop(); }

bool ControlServerWin::start() {
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) return false;

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET) { WSACleanup(); return false; }

    BOOL reuse = TRUE;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);   // loopback only: never reachable from the network

    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(s, 8) != 0) {
        std::cerr << "NAM Capture native control server: impossibile aprire 127.0.0.1:" << port_ << "\n";
        closesocket(s);
        WSACleanup();
        return false;
    }

    listen_socket_ = static_cast<uintptr_t>(s);
    running_ = true;
    accept_thread_ = std::thread(&ControlServerWin::run_loop, this);
    return true;
}

void ControlServerWin::stop() {
    if (!running_.exchange(false)) return;
    closesocket(static_cast<SOCKET>(listen_socket_));   // unblocks accept() in run_loop, which then closes engine_
    if (accept_thread_.joinable()) accept_thread_.join();   // waits for that close to actually finish (fade-out, ASIOExit)
    WSACleanup();
}

void ControlServerWin::run_loop() {
    // COM apartment state is per-thread, and the ASIO SDK's driver COM object (loaded via CoCreateInstance in
    // AudioEngineWin::open, see host/pc/asiolist.cpp) is NOT safe to touch from a different thread than the one
    // that created it without marshaling - which most ASIO driver COM servers don't support. So every request is
    // handled serially, right here on this one thread (true anyway: the Python relay is the only caller, never
    // concurrent) - not dispatched to a fresh thread per connection as ControlServer.swift does on Mac, where
    // CoreAudio has no such constraint. CoInitializeEx once for this thread's whole lifetime, matching that.
    HRESULT com_init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    while (running_) {
        SOCKET client = accept(static_cast<SOCKET>(listen_socket_), nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;   // includes the closesocket()-from-stop() case, which then sees !running_
        handle_client(static_cast<uintptr_t>(client));
    }
    engine_.close();   // same thread that opened it (see above) - never called from stop() directly, on purpose
    if (SUCCEEDED(com_init)) CoUninitialize();
}

void ControlServerWin::handle_client(uintptr_t client_socket) {
    SOCKET client = static_cast<SOCKET>(client_socket);
    std::string raw;
    char buf[4096];
    size_t header_end = std::string::npos;
    while (header_end == std::string::npos) {
        int n = recv(client, buf, sizeof(buf), 0);
        if (n <= 0) { closesocket(client); return; }
        raw.append(buf, n);
        header_end = raw.find("\r\n\r\n");
        if (raw.size() > (1 << 20)) { closesocket(client); return; }   // a header this large is not one of ours
    }

    std::string header_text = raw.substr(0, header_end);
    std::istringstream header_stream(header_text);
    std::string request_line;
    std::getline(header_stream, request_line);
    if (!request_line.empty() && request_line.back() == '\r') request_line.pop_back();
    std::istringstream request_line_stream(request_line);
    std::string method, path;
    request_line_stream >> method >> path;

    long content_length = 0;
    std::string line;
    while (std::getline(header_stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        for (auto& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (key == "content-length") content_length = std::strtol(line.c_str() + colon + 1, nullptr, 10);
    }

    std::string body = raw.substr(header_end + 4);
    while (static_cast<long>(body.size()) < content_length) {
        int n = recv(client, buf, static_cast<int>(std::min<size_t>(sizeof(buf), content_length - body.size())), 0);
        if (n <= 0) break;
        body.append(buf, n);
    }

    int status = 200;
    std::string response_body = route(method, path, body, &status);

    static const char* status_text_200 = "200 OK";
    static const char* status_text_400 = "400 Bad Request";
    static const char* status_text_404 = "404 Not Found";
    const char* status_text = status == 200 ? status_text_200 : (status == 404 ? status_text_404 : status_text_400);

    std::ostringstream response;
    response << "HTTP/1.1 " << status_text << "\r\n"
             << "Content-Type: application/json\r\n"
             << "Content-Length: " << response_body.size() << "\r\n"
             << "Connection: close\r\n\r\n"
             << response_body;
    std::string response_str = response.str();
    send(client, response_str.data(), static_cast<int>(response_str.size()), 0);
    closesocket(client);
}

std::string ControlServerWin::route(const std::string& method, const std::string& path, const std::string& body_json,
                                     int* out_status) {
    json body;
    try { body = body_json.empty() ? json::object() : json::parse(body_json); } catch (...) { body = json::object(); }

    auto ok = [&](json j) { *out_status = 200; j["ok"] = true; return j.dump(); };
    auto bad = [&](const std::string& error) { *out_status = 400; return json{{"ok", false}, {"error", error}}.dump(); };

    if (method == "POST" && path == "/open") {
        if (!body.contains("device") || !body.contains("in_channel") || !body.contains("out_channel"))
            return bad("missing device/in_channel/out_channel");
        std::string err = engine_.open(body["device"].get<std::string>(), body["in_channel"].get<int>(),
                                        body["out_channel"].get<int>(), json_int_array(body, "extra_out_channels"),
                                        body.value("blocksize", 0), body.value("gain_db", 0.0), body.value("ceiling_db", -1.0));
        if (!err.empty()) return bad(err);
        return ok({{"blocksize", engine_.granted_blocksize()}, {"sample_rate", engine_.sample_rate()}});
    }
    if (method == "POST" && path == "/open_dual") {
        if (!body.contains("device") || !body.contains("guitar_channel") || !body.contains("return_channel") ||
            !body.contains("pedal_out_channel") || !body.contains("monitor_out_channels"))
            return bad("missing device/guitar_channel/return_channel/pedal_out_channel/monitor_out_channels");
        std::string err = engine_.open_dual(body["device"].get<std::string>(), body["guitar_channel"].get<int>(),
                                             body["return_channel"].get<int>(), body["pedal_out_channel"].get<int>(),
                                             json_int_array(body, "monitor_out_channels"), body.value("blocksize", 0),
                                             body.value("guitar_gain_db", 0.0), body.value("ceiling_db", -1.0));
        if (!err.empty()) return bad(err);
        return ok({{"blocksize", engine_.granted_blocksize()}, {"sample_rate", engine_.sample_rate()}});
    }
    if (method == "POST" && path == "/open_player") {
        if (!body.contains("device") || !body.contains("guitar_channel") || !body.contains("return_channel") ||
            !body.contains("reamp_out_channel") || !body.contains("monitor_out_channels"))
            return bad("missing device/guitar_channel/return_channel/reamp_out_channel/monitor_out_channels");
        std::string err = engine_.open_player(body["device"].get<std::string>(), body["guitar_channel"].get<int>(),
                                               body["return_channel"].get<int>(), body["reamp_out_channel"].get<int>(),
                                               json_int_array(body, "monitor_out_channels"), body.value("blocksize", 0),
                                               body.value("feed_gain_db", 0.0), body.value("ceiling_db", -1.0));
        if (!err.empty()) return bad(err);
        return ok({{"blocksize", engine_.granted_blocksize()}, {"sample_rate", engine_.sample_rate()}});
    }
    if (method == "POST" && path == "/leg") {            // the A/B switch: "A" = the real gear, "B" = the model
        if (!body.contains("leg") || !body["leg"].is_string()) return bad("missing leg (A or B)");
        const std::string leg = body["leg"].get<std::string>();
        if (leg != "A" && leg != "B") return bad("leg must be A or B");
        engine_.select_leg(leg == "B" ? 1 : 0);
        return ok({{"leg", leg}});
    }
    if (method == "POST" && path == "/amp_gain") {
        if (!body.contains("gain_db")) return bad("missing gain_db");
        engine_.set_amp_gain_db(body["gain_db"].get<double>());
        return ok({{"gain_db", body["gain_db"]}});
    }
    if (method == "POST" && path == "/b_trim") {
        if (!body.contains("trim_db")) return bad("missing trim_db");
        engine_.set_b_trim_db(body["trim_db"].get<double>());
        return ok({{"trim_db", body["trim_db"]}});
    }
    if (method == "POST" && path == "/b_delay") {
        if (!body.contains("samples")) return bad("missing samples");
        engine_.set_b_delay(body["samples"].get<int>());
        return ok({{"samples", body["samples"]}});
    }
    if (method == "POST" && path == "/gain") {
        if (!body.contains("gain_db")) return bad("missing gain_db");
        engine_.set_gain_db(body["gain_db"].get<double>());
        return ok({{"gain_db", body["gain_db"]}});
    }
    if (method == "POST" && path == "/pregain") {
        if (!body.contains("pre_gain_db")) return bad("missing pre_gain_db");
        engine_.set_pre_gain_db(body["pre_gain_db"].get<double>());
        return ok({{"pre_gain_db", body["pre_gain_db"]}});
    }
    if (method == "POST" && path == "/guitar_gain") {
        if (!body.contains("gain_db")) return bad("missing gain_db");
        engine_.set_guitar_gain_db(body["gain_db"].get<double>());
        return ok({{"gain_db", body["gain_db"]}});
    }
    if (method == "POST" && path == "/gate") {
        if (!body.contains("threshold_db")) return bad("missing threshold_db");
        engine_.set_gate_threshold_db(body["threshold_db"].get<double>());
        return ok({{"threshold_db", body["threshold_db"]}});
    }
    if (method == "POST" && path == "/ir_gain") {
        if (body.contains("in_db")) engine_.set_ir_in_gain_db(body["in_db"].get<double>());
        if (body.contains("out_db")) engine_.set_ir_out_gain_db(body["out_db"].get<double>());
        return ok({});
    }
    if (method == "POST" && path == "/eq") {
        if (body.contains("bass_db")) engine_.set_eq_bass_db(body["bass_db"].get<double>());
        if (body.contains("mid_db")) engine_.set_eq_mid_db(body["mid_db"].get<double>());
        if (body.contains("treble_db")) engine_.set_eq_treble_db(body["treble_db"].get<double>());
        return ok({});
    }
    if (method == "POST" && path == "/model") {
        if (!body.contains("path")) return bad("missing path");
        std::string err = engine_.load_model(body["path"].get<std::string>(), legs_mask(body));
        if (!err.empty()) return bad(err);
        AudioEngineWin::State s = engine_.state();
        json j = json::object();
        j["input_level_dbu"] = s.model_input_level_dbu.has_value() ? json(*s.model_input_level_dbu) : json(nullptr);
        j["output_level_dbu"] = s.model_output_level_dbu.has_value() ? json(*s.model_output_level_dbu) : json(nullptr);
        j["sizes"] = engine_.model_sizes();
        return ok(j);
    }
    if (method == "POST" && path == "/model/unload") { engine_.unload_model(legs_mask(body)); return ok({}); }
    if (method == "POST" && path == "/pedal_model") {
        if (!body.contains("path")) return bad("missing path");
        std::string err = engine_.load_pedal(body["path"].get<std::string>());
        if (!err.empty()) return bad(err);
        AudioEngineWin::State s = engine_.state();
        json j = json::object();
        j["input_level_dbu"] = s.pedal_input_level_dbu.has_value() ? json(*s.pedal_input_level_dbu) : json(nullptr);
        j["output_level_dbu"] = s.pedal_output_level_dbu.has_value() ? json(*s.pedal_output_level_dbu) : json(nullptr);
        j["sizes"] = engine_.pedal_sizes();
        return ok(j);
    }
    if (method == "POST" && path == "/pedal_model/unload") { engine_.unload_pedal(); return ok({}); }
    if (method == "POST" && path == "/pedal_controls") {
        if (body.contains("pre_db")) engine_.set_pedal_pre_gain_db(body["pre_db"].get<double>());
        if (body.contains("post_db")) engine_.set_pedal_gain_db(body["post_db"].get<double>());
        int size = -1;
        if (body.contains("size")) size = engine_.set_pedal_size(body["size"].get<int>());
        return ok({{"size", size}, {"sizes", engine_.pedal_sizes()}});
    }
    if (method == "POST" && path == "/amp_pregain") {
        if (!body.contains("gain_db")) return bad("missing gain_db");
        engine_.set_amp_pre_gain_db(body["gain_db"].get<double>());
        return ok({{"gain_db", body["gain_db"]}});
    }
    if (method == "POST" && path == "/model_size") {
        if (!body.contains("index")) return bad("missing index");
        const int index = engine_.set_model_size(body["index"].get<int>());
        return ok({{"index", index}, {"sizes", engine_.model_sizes()}});
    }
    if (method == "POST" && path == "/ir") {
        if (!body.contains("path")) return bad("missing path");
        std::string err = engine_.load_ir(body["path"].get<std::string>(), legs_mask(body));
        if (!err.empty()) return bad(err);
        return ok({});
    }
    if (method == "POST" && path == "/ir/unload") { engine_.unload_ir(legs_mask(body)); return ok({}); }
    if (method == "POST" && path == "/stop") { engine_.close(); return ok({}); }
    if (method == "GET" && path == "/state") {
        AudioEngineWin::State s = engine_.state();
        json j;
        j["opened"] = engine_.is_open();
        j["in_peak_dbfs"] = s.in_peak_dbfs;
        j["out_peak_dbfs"] = s.out_peak_dbfs;
        j["xruns"] = s.xruns;
        j["blocksize"] = engine_.granted_blocksize();
        j["sample_rate"] = engine_.sample_rate();
        j["model_loaded"] = s.model_loaded;
        j["ir_loaded"] = s.ir_loaded;
        j["model_input_level_dbu"] = s.model_input_level_dbu.has_value() ? json(*s.model_input_level_dbu) : json(nullptr);
        j["model_output_level_dbu"] = s.model_output_level_dbu.has_value() ? json(*s.model_output_level_dbu) : json(nullptr);
        j["guitar_in_peak_dbfs"] = s.guitar_in_peak_dbfs.has_value() ? json(*s.guitar_in_peak_dbfs) : json(nullptr);
        j["guitar_out_peak_dbfs"] = s.guitar_out_peak_dbfs.has_value() ? json(*s.guitar_out_peak_dbfs) : json(nullptr);
        j["tuner_hz"] = s.tuner_hz.has_value() ? json(*s.tuner_hz) : json(nullptr);
        j["tuner_confidence"] = s.tuner_confidence.has_value() ? json(*s.tuner_confidence) : json(nullptr);
        j["player"] = s.player;
        if (s.player) {
            j["leg"] = s.leg == 0 ? "A" : "B";
            j["ir_a_loaded"] = s.ir_a_loaded;
            j["a_peak_dbfs"] = s.a_peak_dbfs.has_value() ? json(*s.a_peak_dbfs) : json(nullptr);
            j["b_peak_dbfs"] = s.b_peak_dbfs.has_value() ? json(*s.b_peak_dbfs) : json(nullptr);
            j["pedal_loaded"] = s.pedal_loaded;
        }
        *out_status = 200;
        return j.dump();
    }
    *out_status = 404;
    return json{{"error", "unknown route"}}.dump();
}
