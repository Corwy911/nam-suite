// A tiny, self-contained WAV reader for loading cabinet IRs (see AudioEngineWin::load_ir). Windows has no
// AVFoundation equivalent readily at hand for a plain C++ static lib, and an IR is always a plain WAV (the manual's
// own chapter 6: "IRs must have the interface's own rate: they are never resampled" - so this needs no format
// conversion beyond decoding to float and downmixing to mono, exactly what AudioEngine.swift's loadIR() does with
// AVAudioFile/AVAudioPCMBuffer). Handles PCM 16/24/32-bit and IEEE float WAVs; no compressed formats, no chunks
// beyond fmt/data (WAVE_FORMAT_EXTENSIBLE's sub-format is read from the extended fmt chunk when present).
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct WavData {
    std::vector<float> mono;   // downmixed to mono, in [-1, 1] (clipping not re-checked here: the source file is trusted)
    double sample_rate = 0.0;
};

inline WavData read_wav_mono(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("impossibile aprire il file: " + path);

    char riff[4], wave[4];
    f.read(riff, 4);
    uint32_t riff_size = 0;
    f.read(reinterpret_cast<char*>(&riff_size), 4);
    f.read(wave, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0)
        throw std::runtime_error("non e' un file WAV valido: " + path);

    uint16_t format_tag = 0, channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    bool have_fmt = false;
    std::vector<uint8_t> data_bytes;
    bool have_data = false;

    while (f.good() && !(have_fmt && have_data)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        f.read(chunk_id, 4);
        f.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (!f.good()) break;

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            std::vector<char> fmt(chunk_size);
            f.read(fmt.data(), chunk_size);
            std::memcpy(&format_tag, fmt.data() + 0, 2);
            std::memcpy(&channels, fmt.data() + 2, 2);
            std::memcpy(&sample_rate, fmt.data() + 4, 4);
            std::memcpy(&bits_per_sample, fmt.data() + 14, 2);
            if (format_tag == 0xFFFE && chunk_size >= 40) {   // WAVE_FORMAT_EXTENSIBLE: real tag is in the sub-format GUID's first two bytes
                uint16_t real_tag = 0;
                std::memcpy(&real_tag, fmt.data() + 24, 2);
                format_tag = real_tag;
            }
            have_fmt = true;
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            data_bytes.resize(chunk_size);
            f.read(reinterpret_cast<char*>(data_bytes.data()), chunk_size);
            have_data = true;
        } else {
            f.seekg(chunk_size, std::ios::cur);   // skip chunks we don't need (LIST, fact, cue, ...)
        }
        if (chunk_size % 2 == 1) f.seekg(1, std::ios::cur);   // chunks are word-aligned
    }
    if (!have_fmt || !have_data) throw std::runtime_error("WAV senza chunk fmt/data: " + path);
    if (channels == 0) throw std::runtime_error("WAV con 0 canali: " + path);

    const int bytes_per_sample = bits_per_sample / 8;
    const size_t n_frames = data_bytes.size() / (size_t(bytes_per_sample) * channels);

    WavData out;
    out.sample_rate = static_cast<double>(sample_rate);
    out.mono.resize(n_frames);

    auto read_sample = [&](const uint8_t* p) -> float {
        if (format_tag == 3) {   // IEEE float
            if (bits_per_sample == 32) { float v; std::memcpy(&v, p, 4); return v; }
            if (bits_per_sample == 64) { double v; std::memcpy(&v, p, 8); return static_cast<float>(v); }
        } else if (format_tag == 1) {   // PCM integer
            if (bits_per_sample == 16) { int16_t v; std::memcpy(&v, p, 2); return v / 32768.0f; }
            if (bits_per_sample == 24) {
                int32_t v = (p[0]) | (p[1] << 8) | (p[2] << 16);
                if (v & 0x800000) v |= 0xFF000000;   // sign-extend
                return v / 8388608.0f;
            }
            if (bits_per_sample == 32) { int32_t v; std::memcpy(&v, p, 4); return v / 2147483648.0f; }
        }
        throw std::runtime_error("formato WAV non supportato (tag " + std::to_string(format_tag) +
                                  ", " + std::to_string(bits_per_sample) + " bit)");
    };

    for (size_t i = 0; i < n_frames; ++i) {
        float sum = 0.0f;
        const uint8_t* frame = data_bytes.data() + i * bytes_per_sample * channels;
        for (uint16_t c = 0; c < channels; ++c) sum += read_sample(frame + c * bytes_per_sample);
        out.mono[i] = sum / static_cast<float>(channels);
    }
    return out;
}
