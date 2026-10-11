// The few things that differ between Windows and macOS. Everything else in this program is shared.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;

namespace NamSuite;

/// <summary>Where the installed files are (see suite/installer/NAMSuite.iss and suite/macos/build_mac.sh).</summary>
public sealed class Layout
{
    public string Root;          // folder with nam-web\ and trainers\
    public string PythonExe;
    public string Site;          // packages (PYTHONPATH), or null (development layout: a venv)
    public string DataDir;       // NAM_DATA_DIR for the server, or null = data next to the program (the Windows layout)
}

public interface IPlatform
{
    Layout FindLayout();
    void KeepChildAlive(Process child);                 // make the OS kill the child if this program is killed
    bool AutostartOn();
    void SetAutostart(bool on, string exePath);
    void OpenUrl(string url);
    void OpenFolder(string path);
    void Error(string title, string text);
    bool Ask(string title, string text, string yes, string no);   // true = yes (the safe answer, "no", is the default)
    string TempEnvName { get; }                          // TEMP (Windows) / TMPDIR (macOS)
}

public static class Platform
{
    public static readonly IPlatform Current = OperatingSystem.IsWindows() ? new WindowsPlatform() : new MacPlatform();
}

// ------------------------------------------------------------------------------------------------ Windows
[SupportedOSPlatform("windows")]
public sealed class WindowsPlatform : IPlatform
{
    [StructLayout(LayoutKind.Sequential)]
    struct IoCounters { public ulong a, b, c, d, e, f; }
    [StructLayout(LayoutKind.Sequential)]
    struct BasicLimit
    {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    struct ExtendedLimit
    {
        public BasicLimit Basic;
        public IoCounters Io;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll")] static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint size);
    [DllImport("kernel32.dll")] static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int MessageBoxW(IntPtr hwnd, string text, string caption, uint type);

    const string RunKey = @"Software\Microsoft\Windows\CurrentVersion\Run";
    const string RunName = "NAM Trainer";
    static IntPtr job = IntPtr.Zero;

    public string TempEnvName => "TEMP";

    public Layout FindLayout()
    {
        string exeDir = AppContext.BaseDirectory.TrimEnd('\\');
        string root = Environment.GetEnvironmentVariable("NAM_TRAINER_ROOT") is { Length: > 0 } r ? r : exeDir;
        string suitePython = Path.GetFullPath(Path.Combine(root, "..", "runtime", "python.exe"));
        string devPython = Path.Combine(root, "venv", "Scripts", "python.exe");
        string python = Environment.GetEnvironmentVariable("NAM_PYTHON") is { Length: > 0 } p ? p : File.Exists(suitePython) ? suitePython : devPython;
        string site = Path.Combine(root, "site");
        return new Layout { Root = root, PythonExe = python, Site = python == suitePython && Directory.Exists(site) ? site : null,
                            DataDir = Environment.GetEnvironmentVariable("NAM_DATA_DIR") is { Length: > 0 } d ? d : null };
    }

    public void KeepChildAlive(Process child)
    {
        if (job == IntPtr.Zero)
        {
            job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) return;
            var info = new ExtendedLimit();
            info.Basic.LimitFlags = 0x2000;   // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            int size = Marshal.SizeOf<ExtendedLimit>();
            IntPtr mem = Marshal.AllocHGlobal(size);
            try { Marshal.StructureToPtr(info, mem, false); SetInformationJobObject(job, 9, mem, (uint)size); }
            finally { Marshal.FreeHGlobal(mem); }
        }
        AssignProcessToJobObject(job, child.Handle);
    }

    public bool AutostartOn()
    {
        try { using var k = Microsoft.Win32.Registry.CurrentUser.OpenSubKey(RunKey); return k?.GetValue(RunName) != null; }
        catch (Exception) { return false; }
    }

    public void SetAutostart(bool on, string exePath)
    {
        using var k = Microsoft.Win32.Registry.CurrentUser.CreateSubKey(RunKey);
        if (on) k.SetValue(RunName, $"\"{exePath}\" --background"); else k.DeleteValue(RunName, false);
    }

    public void OpenUrl(string url) => Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
    public void OpenFolder(string path) => Process.Start("explorer.exe", $"\"{path}\"");
    public void Error(string title, string text) => MessageBoxW(IntPtr.Zero, text, title, 0x10 /* MB_ICONERROR */);
    public bool Ask(string title, string text, string yes, string no)
        => MessageBoxW(IntPtr.Zero, text, title, 0x4 /* MB_YESNO */ | 0x30 /* warning */ | 0x100 /* default: No */) == 6;
}

// ------------------------------------------------------------------------------------------------ macOS
public sealed class MacPlatform : IPlatform
{
    const string AgentLabel = "com.namsuite.trainer";
    static string AgentPath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Library", "LaunchAgents", AgentLabel + ".plist");

    public string TempEnvName => "TMPDIR";

    public Layout FindLayout()
    {
        // NAM Trainer.app/Contents/MacOS/NamTrainer  ->  Contents/Resources/{python, trainer}
        string macos = AppContext.BaseDirectory.TrimEnd('/');
        string resources = Path.GetFullPath(Path.Combine(macos, "..", "Resources"));
        string root = Environment.GetEnvironmentVariable("NAM_TRAINER_ROOT") is { Length: > 0 } r ? r : Path.Combine(resources, "trainer");
        string bundledPython = Path.Combine(resources, "python", "bin", "python3");
        string python = Environment.GetEnvironmentVariable("NAM_PYTHON") is { Length: > 0 } p ? p : File.Exists(bundledPython) ? bundledPython : Path.Combine(root, "venv", "bin", "python3");
        string site = Path.Combine(root, "site");
        string data = Environment.GetEnvironmentVariable("NAM_DATA_DIR") is { Length: > 0 } d ? d
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Library", "Application Support", "NAM Suite", "Trainer");
        return new Layout { Root = root, PythonExe = python, Site = Directory.Exists(site) ? site : null, DataDir = data };
    }

    public void KeepChildAlive(Process child) { /* the server exits when its parent is gone (NAM_PARENT_PID, see nam-web/app.py) */ }

    public bool AutostartOn() => File.Exists(AgentPath);

    public void SetAutostart(bool on, string exePath)
    {
        if (!on) { File.Delete(AgentPath); return; }
        Directory.CreateDirectory(Path.GetDirectoryName(AgentPath));
        File.WriteAllText(AgentPath, $@"<?xml version=""1.0"" encoding=""UTF-8""?>
<!DOCTYPE plist PUBLIC ""-//Apple//DTD PLIST 1.0//EN"" ""http://www.apple.com/DTDs/PropertyList-1.0.dtd"">
<plist version=""1.0""><dict>
  <key>Label</key><string>{AgentLabel}</string>
  <key>ProgramArguments</key><array><string>{System.Security.SecurityElement.Escape(exePath)}</string><string>--background</string></array>
  <key>RunAtLoad</key><true/>
</dict></plist>
", new UTF8Encoding(false));
    }

    static void Run(string file, params string[] args)
    {
        var psi = new ProcessStartInfo(file) { UseShellExecute = false, CreateNoWindow = true };
        foreach (string a in args) psi.ArgumentList.Add(a);
        Process.Start(psi);
    }

    public void OpenUrl(string url) => Run("/usr/bin/open", url);
    public void OpenFolder(string path) => Run("/usr/bin/open", path);

    static string Script(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");

    public void Error(string title, string text)
        => Run("/usr/bin/osascript", "-e", $"display dialog \"{Script(text)}\" with title \"{Script(title)}\" buttons {{\"OK\"}} default button 1 with icon stop");

    public bool Ask(string title, string text, string yes, string no)
    {
        var psi = new ProcessStartInfo("/usr/bin/osascript") { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true };
        psi.ArgumentList.Add("-e");
        psi.ArgumentList.Add($"button returned of (display dialog \"{Script(text)}\" with title \"{Script(title)}\" buttons {{\"{Script(no)}\", \"{Script(yes)}\"}} default button 1 with icon caution)");
        using var p = Process.Start(psi);
        string answer = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit();
        return answer == yes;
    }
}
