// The icon and its menu.
using System.Diagnostics;
using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Platform;
using Avalonia.Themes.Fluent;
using Avalonia.Threading;

namespace NamSuite;

public sealed class App : Application
{
    TrayIcon tray;
    NativeMenuItem autostart;
    DispatcherTimer poll;
    IClassicDesktopStyleApplicationLifetime lifetime;
    bool stopping;
    int restarts;

    public override void Initialize()
    {
        Styles.Add(new FluentTheme());
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            lifetime = desktop;
            desktop.ShutdownMode = ShutdownMode.OnExplicitShutdown;
            BuildTray();
            if (Program.OpenPageAtStart) OpenPage();
        }
        base.OnFrameworkInitializationCompleted();
    }

    static TrainerServer Server => Program.Server;
    static string T(string it, string en) => Server.T(it, en);

    void BuildTray()
    {
        var menu = new NativeMenu();
        menu.Items.Add(Item(T("Apri NAM Trainer", "Open NAM Trainer"), OpenPage));
        menu.Items.Add(new NativeMenuItemSeparator());
        menu.Items.Add(Item(T("Cartella dei modelli", "Models folder"), () => OpenFolder("NAM Generati")));
        menu.Items.Add(Item(T("Cartella dei WAV", "WAV files folder"), () => OpenFolder("File WAV caricati")));
        menu.Items.Add(new NativeMenuItemSeparator());
        autostart = new NativeMenuItem(OperatingSystem.IsWindows() ? T("Avvia con Windows", "Start with Windows") : T("Avvia al login", "Start at login"))
        {
            ToggleType = NativeMenuItemToggleType.CheckBox, IsChecked = Platform.Current.AutostartOn(),
        };
        autostart.Click += (_, _) => ToggleAutostart();
        menu.Items.Add(autostart);
        menu.Items.Add(new NativeMenuItemSeparator());
        menu.Items.Add(Item(T("Esci", "Quit"), Quit));

        tray = new TrayIcon
        {
            Icon = new WindowIcon(AssetLoader.Open(new Uri("avares://NamTrainer/Assets/tray.png"))),
            ToolTipText = "NAM Trainer",
            Menu = menu,
            IsVisible = true,
        };
        tray.Clicked += (_, _) => { if (OperatingSystem.IsWindows()) OpenPage(); };   // on macOS a click opens the menu
        TrayIcon.SetIcons(this, new TrayIcons { tray });

        poll = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
        int ticks = 0;
        poll.Tick += (_, _) =>
        {
            foreach (string command in Program.Instance.Take())            // other launches of the program ask for things through a file
            {
                if (command == "open") OpenPage();
                else if (command == "quit") Shutdown();
            }
            if (++ticks % 6 == 0) Refresh();                               // every 3 seconds
        };
        poll.Start();
    }

    static NativeMenuItem Item(string header, Action action)
    {
        var item = new NativeMenuItem(header);
        item.Click += (_, _) => action();
        return item;
    }

    void OpenPage()
    {
        try { Platform.Current.OpenUrl(Server.Url); } catch (Exception) { }
    }

    void OpenFolder(string name)
    {
        string path = Path.Combine(Server.Layout.DataDir ?? Server.Layout.Root, name);
        try { Directory.CreateDirectory(path); Platform.Current.OpenFolder(path); } catch (Exception) { }
    }

    void ToggleAutostart()
    {
        try { Platform.Current.SetAutostart(autostart.IsChecked, Environment.ProcessPath); }
        catch (Exception e) { Platform.Current.Error("NAM Trainer", e.Message); }
        autostart.IsChecked = Platform.Current.AutostartOn();
    }

    void Refresh()
    {
        if (stopping) return;
        if (Server.HasStopped)
        {
            if (restarts < 2 && Server.Start(out _)) { restarts++; return; }
            tray.ToolTipText = "NAM Trainer - " + T("fermo", "stopped");
            return;
        }
        Task.Run(Server.ReadStatus).ContinueWith(t =>
            Dispatcher.UIThread.Post(() =>
            {
                if (stopping || t.IsFaulted) return;
                Status s = t.Result;
                string text = "NAM Trainer";
                if (s.Running != null) text += " - training " + (s.Percent > 0 ? Math.Round(s.Percent).ToString(CultureInfo.InvariantCulture) + "%" : "…");
                else if (s.Queued > 0) text += " - " + T("in coda: ", "queued: ") + s.Queued;
                tray.ToolTipText = text;
            }));
    }

    void Quit()
    {
        Status s = Server.ReadStatus();
        if (s.Running != null || s.Queued > 0)
        {
            string what = s.Running != null ? T($"È in corso un training ({s.Running}).", $"A training is running ({s.Running}).") : T("Ci sono dei training in coda.", "Trainings are queued.");
            if (!Platform.Current.Ask("NAM Trainer", what + "\n" + T("Uscendo vengono interrotti e dovrai rifarli. Uscire comunque?", "Quitting stops them and you will have to run them again. Quit anyway?"),
                    T("Esci", "Quit"), T("Annulla", "Cancel")))
                return;
        }
        Shutdown();
    }

    void Shutdown()
    {
        if (stopping) return;
        stopping = true;
        poll?.Stop();
        if (tray != null) tray.IsVisible = false;
        Server.Stop();
        lifetime?.Shutdown();
    }
}
