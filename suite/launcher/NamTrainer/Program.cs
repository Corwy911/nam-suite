// NAM Trainer launcher: keeps the trainer's server running, with an icon in the notification area (Windows) / menu bar (macOS).
//
//   NamTrainer                 start the server (hidden), show the icon and open the page; stays until "Quit" in the icon's menu
//   NamTrainer (again)         only opens the page of the running one
//   --background               like the first, but without opening the page (used by "Start with Windows / at login")
//   --no-browser               same without opening the page
//   --quit                     close a running one and wait until its server is gone (used by the installer / uninstaller)
//   --port N                   default 8765
// Environment (tests, development): NAM_TRAINER_ROOT (folder with nam-web), NAM_PYTHON, NAM_DATA_DIR.
using System.Globalization;
using Avalonia;

namespace NamSuite;

public static class Program
{
    public static TrainerServer Server;
    public static bool OpenPageAtStart;
    public static Instance Instance;

    [STAThread]
    public static int Main(string[] args)
    {
        int port = 8765;
        bool noBrowser = false, background = false, quit = false, portGiven = false;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--port" && i + 1 < args.Length) { int.TryParse(args[++i], NumberStyles.None, CultureInfo.InvariantCulture, out port); portGiven = true; }
            else if (args[i] == "--no-browser") noBrowser = true;
            else if (args[i] == "--background") background = true;
            else if (args[i] == "--quit") quit = true;
        }

        var layout = Platform.Current.FindLayout();
        Server = new TrainerServer(port, layout);
        Instance = new Instance(Path.Combine(Server.StateDir, "launcher"), port);

        if (quit)                                           // close the running one(s) (if any) and wait for them
        {
            // without --port: every launcher of this installation, whatever its port (the installer and uninstaller do not know it)
            var instances = new List<Instance>();
            if (portGiven) instances.Add(Instance);
            else foreach (int running in Instance.PortsInUse(Path.Combine(Server.StateDir, "launcher"))) instances.Add(new Instance(Path.Combine(Server.StateDir, "launcher"), running));
            foreach (Instance one in instances)
            {
                if (one.IsFree()) continue;
                one.Send("quit");
                for (int i = 0; i < 60 && !one.IsFree(); i++) Thread.Sleep(250);
                Thread.Sleep(1000);                         // the lock is released a moment before the process is really gone
            }
            return 0;
        }
        if (!Instance.TryBecomeOwner())                     // already running: ask it to show the page
        {
            if (!background) Instance.Send("open");
            return 0;
        }
        if (!Server.Start(out string error))
        {
            Platform.Current.Error("NAM Trainer", error);
            Server.Stop();
            return 1;
        }
        OpenPageAtStart = !background && !noBrowser;
        try
        {
            return BuildAvaloniaApp().StartWithClassicDesktopLifetime(args, Avalonia.Controls.ShutdownMode.OnExplicitShutdown);
        }
        finally
        {
            Server.Stop();
            Instance.Dispose();
        }
    }

    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<App>().UsePlatformDetect();
}

/// <summary>One program per user and port: a lock file held while running, and a small command file the others write into.</summary>
public sealed class Instance : IDisposable
{
    readonly string lockPath, cmdPath;
    FileStream held;

    public Instance(string folder, int port)
    {
        Directory.CreateDirectory(folder);
        lockPath = Path.Combine(folder, $"instance-{port}.lock");
        cmdPath = Path.Combine(folder, $"command-{port}.txt");
    }

    /// <summary>Ports of the launchers that currently hold their lock in this folder.</summary>
    public static IEnumerable<int> PortsInUse(string folder)
    {
        if (!Directory.Exists(folder)) yield break;
        foreach (string file in Directory.GetFiles(folder, "instance-*.lock"))
        {
            string name = Path.GetFileNameWithoutExtension(file);
            if (int.TryParse(name.Substring("instance-".Length), NumberStyles.None, CultureInfo.InvariantCulture, out int port)) yield return port;
        }
    }

    public bool TryBecomeOwner()
    {
        try { held = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); return true; }
        catch (IOException) { return false; }
    }

    public bool IsFree()
    {
        try { using var f = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None); return true; }
        catch (IOException) { return false; }
    }

    public void Send(string command)
    {
        try { File.AppendAllText(cmdPath, command + "\n"); } catch (IOException) { }
    }

    /// <summary>Commands written by other launches since the last call.</summary>
    public string[] Take()
    {
        try
        {
            if (!File.Exists(cmdPath)) return Array.Empty<string>();
            string[] lines = File.ReadAllLines(cmdPath);
            File.Delete(cmdPath);
            return lines;
        }
        catch (IOException) { return Array.Empty<string>(); }
    }

    public void Dispose()
    {
        held?.Dispose();
        held = null;
    }
}
