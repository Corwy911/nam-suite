// Starting, recognising, watching and stopping the trainer's web server (nam-web/app.py).
using System.Diagnostics;
using System.Globalization;
using System.Net.Http;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace NamSuite;

public sealed class Status
{
    public bool Reachable;
    public string Running;       // name of the training in progress, or null
    public double Percent;       // its progress
    public int Queued;
}

public sealed class TrainerServer
{
    public readonly int Port;
    public readonly Layout Layout;
    public readonly string StateDir;       // settings.json, hardware.json, tmp, cache, logs
    public readonly string Library;        // this installation's "NAM Generati" folder (how its own server is recognised)
    public readonly string Language;       // "it" | "en"
    public string OtherLibrary;            // set when another copy of NAM Trainer holds the port

    Process proc;
    bool owns;
    static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };

    public TrainerServer(int port, Layout layout)
    {
        Port = port;
        Layout = layout;
        string data = layout.DataDir ?? layout.Root;
        StateDir = layout.DataDir != null ? Path.Combine(layout.DataDir, "state") : Path.Combine(layout.Root, "nam-web");
        Library = Path.GetFullPath(Path.Combine(data, "NAM Generati"));
        Language = ReadLanguage();
    }

    string L(string it, string en) => Language == "it" ? it : en;
    public string T(string it, string en) => L(it, en);
    public string Url => $"http://localhost:{Port}/";
    public string WebDir => Path.Combine(Layout.Root, "nam-web");

    string ReadLanguage()
    {
        foreach (string file in new[] { Path.Combine(StateDir, "hardware.json"), Path.Combine(Layout.Root, "nam-web", "hardware.json") })
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(file, Encoding.UTF8));
                if (doc.RootElement.TryGetProperty("language", out var v) && v.GetString() is "it" or "en") return v.GetString();
            }
            catch (Exception) { }
        }
        return CultureInfo.CurrentUICulture.TwoLetterISOLanguageName == "it" ? "it" : "en";
    }

    string Get(string path)
    {
        try { return Http.GetStringAsync($"http://127.0.0.1:{Port}{path}").GetAwaiter().GetResult(); }
        catch (Exception) { return null; }
    }

    /// <summary>0 nobody (or not NAM Trainer), 1 THIS installation's server, 2 another copy of NAM Trainer.</summary>
    public int Kind()
    {
        string body = Get("/api/config");
        if (body == null || !body.Contains("\"gear_types\"")) return 0;
        try
        {
            using var doc = JsonDocument.Parse(body);
            string lib = doc.RootElement.GetProperty("paths").GetProperty("library").GetString().TrimEnd(Path.DirectorySeparatorChar);
            if (string.Equals(lib, Library.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase)) return 1;
            OtherLibrary = lib;
            return 2;
        }
        catch (Exception) { return 0; }
    }

    public bool IsMine() => Kind() == 1;

    bool PortBusy()
    {
        try
        {
            using var c = new TcpClient();
            return c.ConnectAsync("127.0.0.1", Port).Wait(700) && c.Connected;
        }
        catch (Exception) { return false; }
    }

    public bool Start(out string error)
    {
        error = null;
        int kind = Kind();
        if (kind == 1) { owns = false; return true; }         // this installation's server already runs: just use it
        if (kind == 2)                                        // another copy holds the port: never attach to it (wrong models folder)
        {
            error = L($"La porta {Port} è occupata da un'altra copia di NAM Trainer (modelli in: {OtherLibrary}).\nChiudila (per esempio con il suo \"Ferma NAM.bat\") e riapri NAM Trainer.",
                      $"Port {Port} is used by another copy of NAM Trainer (models in: {OtherLibrary}).\nClose it (for example with its \"Ferma NAM.bat\") and open NAM Trainer again.");
            return false;
        }
        if (PortBusy()) { error = L($"La porta {Port} è già usata da un altro programma.", $"Port {Port} is already used by another program."); return false; }
        if (!File.Exists(Layout.PythonExe)) { error = L($"Python non trovato: {Layout.PythonExe}", $"Python not found: {Layout.PythonExe}"); return false; }

        string cache = Path.Combine(StateDir, "cache"), tmp = Path.Combine(StateDir, "tmp");
        Directory.CreateDirectory(cache);
        Directory.CreateDirectory(tmp);
        var psi = new ProcessStartInfo(Layout.PythonExe)
        {
            WorkingDirectory = WebDir, UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
        };
        psi.ArgumentList.Add(Path.Combine(WebDir, "app.py"));
        psi.ArgumentList.Add("--port");
        psi.ArgumentList.Add(Port.ToString(CultureInfo.InvariantCulture));
        var env = psi.Environment;
        env.Remove("PYTHONHOME");
        env["PYTHONUTF8"] = "1";
        env["PYTHONUNBUFFERED"] = "1";
        env["PYTHONDONTWRITEBYTECODE"] = "1";
        if (Layout.Site != null) env["PYTHONPATH"] = Layout.Site; else env.Remove("PYTHONPATH");
        // Nothing may be written outside the installation / data folder (the uninstaller must leave no trace): caches and temp files of the
        // libraries go under the state folder, next to what app.py already keeps there.
        env["XDG_CACHE_HOME"] = cache;
        env["HF_HOME"] = Path.Combine(cache, "huggingface");
        env["TORCH_HOME"] = Path.Combine(cache, "torch");
        env["NUMBA_CACHE_DIR"] = Path.Combine(cache, "numba");
        env["MPLCONFIGDIR"] = Path.Combine(cache, "matplotlib");
        env["TORCHINDUCTOR_CACHE_DIR"] = Path.Combine(cache, "inductor");
        env[Platform.Current.TempEnvName] = OperatingSystem.IsWindows() ? tmp : tmp + "/";
        if (OperatingSystem.IsWindows()) env["TMP"] = tmp;
        if (Layout.DataDir != null) env["NAM_DATA_DIR"] = Layout.DataDir;
        env["NAM_PARENT_PID"] = Environment.ProcessId.ToString(CultureInfo.InvariantCulture);
        try { proc = Process.Start(psi); }
        catch (Exception e) { error = e.Message; return false; }
        owns = true;
        Platform.Current.KeepChildAlive(proc);
        Pump(proc.StandardOutput, "server.out.log");
        Pump(proc.StandardError, "server.err.log");
        for (int i = 0; i < 240; i++)                         // up to 60 s: the first start imports PyTorch
        {
            if (proc.HasExited) { error = L("Il server si è fermato subito (vedi server.err.log nella cartella di stato).", "The server stopped right away (see server.err.log in the state folder)."); return false; }
            if (IsMine()) return true;
            Thread.Sleep(250);
        }
        error = L("Il server non risponde (vedi server.err.log nella cartella di stato).", "The server does not answer (see server.err.log in the state folder).");
        return false;
    }

    void Pump(StreamReader reader, string fileName)
    {
        string path = Path.Combine(StateDir, fileName);
        try { if (File.Exists(path) && new FileInfo(path).Length > 2 * 1024 * 1024) File.Delete(path); } catch (Exception) { }
        new Thread(() =>
        {
            try
            {
                using var w = new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), new UTF8Encoding(false));
                string line;
                while ((line = reader.ReadLine()) != null) { w.WriteLine(line); w.Flush(); }
            }
            catch (Exception) { }
        }) { IsBackground = true }.Start();
    }

    public bool HasStopped => owns && proc != null && proc.HasExited;

    public void Stop()
    {
        try
        {
            if (owns && proc != null && !proc.HasExited)
            {
                proc.Kill(entireProcessTree: true);          // the server and the training process it started
                proc.WaitForExit(5000);
            }
        }
        catch (Exception) { }
    }

    public Status ReadStatus()
    {
        var s = new Status();
        string body = Get("/api/models");
        if (body == null) return s;
        s.Reachable = true;
        try
        {
            using var doc = JsonDocument.Parse(body);
            foreach (var m in doc.RootElement.EnumerateArray())
            {
                if (!m.TryGetProperty("state", out var st)) continue;
                string state = st.GetString();
                if (state == "queued") s.Queued++;
                if (state != "running") continue;
                s.Running = m.GetProperty("name").GetString();
                if (m.TryGetProperty("progress", out var p) && p.ValueKind == JsonValueKind.Object
                    && p.TryGetProperty("epochs", out var epochs) && epochs.GetDouble() > 0)
                {
                    double epoch = p.GetProperty("epoch").GetDouble();
                    double batch = p.TryGetProperty("batch", out var b) && b.ValueKind == JsonValueKind.Number ? b.GetDouble() : 0;
                    double batches = p.TryGetProperty("batches", out var bs) && bs.ValueKind == JsonValueKind.Number ? bs.GetDouble() : 0;
                    s.Percent = Math.Min(100, (epoch + (batches > 0 ? batch / batches : 0)) / epochs.GetDouble() * 100);
                }
            }
        }
        catch (Exception) { }
        return s;
    }
}
