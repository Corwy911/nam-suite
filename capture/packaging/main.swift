// NAM Capture - native launcher: a normal AppKit window with a WKWebView, backed by the Python capture server as a
// child process. This is the whole "app" as far as macOS is concerned: it gets its own icon, Dock entry, Cmd+Q, window
// close behaviour - no browser chrome, no address bar. The capture tool's own logic (audio, calibration, the wizard
// page) is unchanged; this file only gives it a window to live in.

import Cocoa
import WebKit
import AVFoundation

let PORT = "8766"

// The per-user data folder: NAM_CAPTURE_DATA (a test run, see the 'test' folder) or the standard one.
func dataDirURL() -> URL {
    if let custom = ProcessInfo.processInfo.environment["NAM_CAPTURE_DATA"], !custom.isEmpty {
        return URL(fileURLWithPath: (custom as NSString).expandingTildeInPath)
    }
    return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/NAM Capture")
}
let BASE_URL = "http://127.0.0.1:\(PORT)/"

final class AppDelegate: NSObject, NSApplicationDelegate, WKUIDelegate, NSMenuDelegate {
    var window: NSWindow!
    var webView: WKWebView!
    var serverProcess: Process?
    var logHandle: FileHandle?
    var trainerStatusItem: NSMenuItem?     // the File menu's own non-clickable status line, see menuWillOpen
    let controlServer = ControlServer()

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildWindow()
        requestMicrophone()            // before the first "Apri la scheda": while the system prompt is still
                                       // pending, PortAudio fails the input side with a generic paInternalError
        startServer()
        controlServer.start()          // the native real-time engine's local control API (127.0.0.1:8767); the
                                        // Python server proxies /api/capture/native/* to it - see capture_api.py
        waitForServer(triesLeft: 60)
        NSApp.activate(ignoringOtherApps: true)
    }

    // macOS gates the input side of every interface behind the microphone permission, and PortAudio's only symptom
    // of a missing one is the generic paInternalError (-9986), indistinguishable from a driver problem. Ask at
    // launch, so the system prompt is answered long before the user reaches "Apri la scheda" - and if it was
    // denied, say so plainly instead of letting the first open fail with a cryptic error.
    func requestMicrophone() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                if !granted { DispatchQueue.main.async { self.microphoneDenied() } }
            }
        case .denied, .restricted:
            microphoneDenied()
        default:
            break
        }
    }

    func microphoneDenied() {
        let alert = NSAlert()
        alert.messageText = "Permesso microfono mancante / Missing microphone permission"
        alert.informativeText = "NAM Capture non può aprire gli ingressi della scheda. Consenti il microfono in Impostazioni sistema → Privacy e sicurezza → Microfono e riapri l'app.\n\nNAM Capture cannot open the interface's inputs. Allow the microphone in System Settings → Privacy & Security → Microphone and reopen the app."
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    func buildWindow() {
        let screen = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1280, height: 900)
        let w = min(1180, screen.width - 80), h = min(880, screen.height - 80)
        let rect = NSRect(x: 0, y: 0, width: w, height: h)
        window = NSWindow(contentRect: rect, styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "NAM Capture"
        window.center()
        window.minSize = NSSize(width: 760, height: 560)

        let config = WKWebViewConfiguration()
        config.preferences.setValue(true, forKey: "developerExtrasEnabled")  // right-click "Inspect Element" while we're building this
        webView = WKWebView(frame: rect, configuration: config)
        webView.autoresizingMask = [.width, .height]
        webView.uiDelegate = self
        webView.loadHTMLString(loadingHTML(), baseURL: nil)
        window.contentView = webView
        window.makeKeyAndOrderFront(nil)
    }

    // A link with target="_blank" (the NAM guide, the trainer's own page) asks the web view for a new window: with no answer it does nothing at
    // all. The page's own address never changes; anything that wants a window goes to the system browser instead.
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction,
                 windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = navigationAction.request.url, url.scheme == "http" || url.scheme == "https" {
            NSWorkspace.shared.open(url)
        }
        return nil
    }

    // WKWebView has no built-in JavaScript dialogs: without these, confirm() silently returns false and every
    // button the page guards with a safety confirm (tone, calibration, monitor, live, capture) does nothing at all.
    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Annulla")
        completionHandler(alert.runModal() == .alertFirstButtonReturn)
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = message
        alert.addButton(withTitle: "OK")
        alert.runModal()
        completionHandler()
    }

    // Without this, <input type="file"> (every "Sfoglia...") behaves as if the user pressed Cancel as soon as a WKUIDelegate exists.
    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters, initiatedByFrame frame: WKFrameInfo,
                 completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        let response = panel.runModal()                 // modal, like the confirm dialogs: every "Sfoglia" opens it, not only the first
        completionHandler(response == .OK ? panel.urls : nil)
    }

    // The trainer hand-off (File menu): a real folder path only ever comes from a native NSOpenPanel here, never from
    // the page's own <input type=file> - the browser's File API never exposes an absolute path for a folder picked
    // that way. The chosen path is handed to the page as a plain JS call, exactly the shape trainerSetFolder/
    // trainerBrowseFolder in app.js expect (see capture_api/trainer.py for what happens with it server-side).
    func callPageFunction(_ name: String, path: String) {
        guard let json = try? JSONSerialization.data(withJSONObject: [path]),
              let args = String(data: json, encoding: .utf8) else { return }   // a 1-element JSON array: a safely escaped call argument
        webView.evaluateJavaScript("window.\(name).apply(null, \(args))")
    }

    // A plain text prompt (no NSOpenPanel involved: an address isn't a filesystem path) - same NSAlert pattern as
    // the JS confirm/alert dialogs above, with a text field as its accessory view.
    @objc func setTrainerHost(_ sender: Any?) {
        let alert = NSAlert()
        alert.messageText = "Indirizzo del NAM Trainer"
        alert.informativeText = "L'indirizzo del PC dove gira il NAM Trainer, quello scritto sotto la pagina del trainer (es. 192.168.1.42). Se è questo stesso PC: localhost."
        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 240, height: 24))
        alert.accessoryView = field
        alert.addButton(withTitle: "Imposta")
        alert.addButton(withTitle: "Annulla")
        alert.window.initialFirstResponder = field
        if alert.runModal() == .alertFirstButtonReturn {
            let host = field.stringValue.trimmingCharacters(in: .whitespaces)
            if !host.isEmpty { callPageFunction("trainerSetHost", path: host) }
        }
    }

    @objc func browseCapturesFolder(_ sender: Any?) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Scegli una cartella con catture già fatte da esportare"
        panel.prompt = "Scegli"
        // always starts from the library's own Catture folder - that's where a finished capture normally lives
        let defaultDir = dataDirURL().appendingPathComponent("File WAV caricati/Catture")
        panel.directoryURL = defaultDir
        if panel.runModal() == .OK, let url = panel.url {
            callPageFunction("trainerBrowseFolder", path: url.path)
        }
    }

    // Refreshes the File menu's status line only when it is actually opened (no background polling) - cheap local
    // GET to the same endpoint the page itself polls (capture_api/trainer.py's GET /trainer/status).
    func menuWillOpen(_ menu: NSMenu) {
        guard let item = trainerStatusItem, menu.items.contains(item) else { return }
        item.title = "NAM Trainer: verifico…"
        var req = URLRequest(url: URL(string: BASE_URL + "api/capture/trainer/status")!)
        req.timeoutInterval = 2.0
        URLSession.shared.dataTask(with: req) { data, _, _ in
            DispatchQueue.main.async {
                guard let data = data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    item.title = "NAM Trainer: server non raggiungibile"
                    return
                }
                guard let host = obj["host"] as? String else {
                    item.title = "NAM Trainer: nessun indirizzo impostato"
                    return
                }
                let online = (obj["online"] as? Bool) ?? false
                item.title = "NAM Trainer (\(host)): " + (online ? "rilevato in rete" : "non rilevato")
            }
        }.resume()
    }

    func loadingHTML(_ message: String = "Avvio del catturatore…") -> String {
        return """
        <html><body style="background:#0f172a;color:#e2e8f0;font:15px -apple-system,system-ui;
        display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
        <div>\(message)</div></body></html>
        """
    }

    func resourcePath() -> String {
        return Bundle.main.resourcePath ?? FileManager.default.currentDirectoryPath
    }

    func startServer() {
        let res = resourcePath()
        // Two layouts: the Suite's (a pinned standalone Python in python/ + its packages in site/) or the development
        // build's own venv/ (see build_app.sh).
        let suitePython = res + "/python/bin/python3"
        let suiteLayout = FileManager.default.fileExists(atPath: suitePython)
        let python = suiteLayout ? suitePython : res + "/venv/bin/python"
        let appPy = res + "/app/app.py"
        guard FileManager.default.fileExists(atPath: python) else {
            webView.loadHTMLString(loadingHTML("Errore: manca " + python + ". Ricompila con packaging/build_app.sh."), baseURL: nil)
            return
        }
        let logDir = dataDirURL()
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        let logURL = logDir.appendingPathComponent("app.log")
        FileManager.default.createFile(atPath: logURL.path, contents: nil)
        logHandle = try? FileHandle(forWritingTo: logURL)

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = [appPy, "--port", PORT]
        proc.currentDirectoryURL = URL(fileURLWithPath: res + "/app")
        if suiteLayout {
            var env = ProcessInfo.processInfo.environment
            env["PYTHONPATH"] = res + "/site"
            env["PYTHONNOUSERSITE"] = "1"
            proc.environment = env
        }
        if let h = logHandle {
            proc.standardOutput = h
            proc.standardError = h
        }
        do {
            try proc.run()
            serverProcess = proc
        } catch {
            webView.loadHTMLString(loadingHTML("Errore all'avvio del server: \(error)"), baseURL: nil)
        }
    }

    func waitForServer(triesLeft: Int) {
        let url = URL(string: BASE_URL + "health")!
        var req = URLRequest(url: url)
        req.timeoutInterval = 1
        let task = URLSession.shared.dataTask(with: req) { _, response, _ in
            DispatchQueue.main.async {
                if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                    self.webView.load(URLRequest(url: URL(string: BASE_URL)!))
                } else if triesLeft > 0 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { self.waitForServer(triesLeft: triesLeft - 1) }
                } else {
                    self.webView.loadHTMLString(self.loadingHTML("Il server non risponde. Controlla ~/Library/Application Support/NAM Capture/app.log"), baseURL: nil)
                }
            }
        }
        task.resume()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        controlServer.stop()
        if let p = serverProcess, p.isRunning {
            p.terminate()             // the server's lifespan handler releases the audio interface on SIGTERM
            p.waitUntilExit()
        }
        try? logHandle?.close()
    }
}

// A minimal main menu bar: without one, Cmd+Q (and every other standard shortcut) has nothing bound to it, since this
// app is built from a plain main.swift rather than a storyboard/Xcode template that would set this up for free.
func buildMainMenu(_ delegate: AppDelegate) -> NSMenu {
    let main = NSMenu()

    let appMenuItem = NSMenuItem()
    main.addItem(appMenuItem)
    let appMenu = NSMenu()
    appMenuItem.submenu = appMenu
    appMenu.addItem(withTitle: "Chiudi NAM Capture", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

    // The trainer hand-off (Roman: "dove di solito ci sono file, vista ecc", not on the page itself) - detected
    // live on the LAN (see capture_api/trainer.py), no shared folder. Every item here needs an explicit target:
    // unlike the standard AppKit selectors below, the responder chain doesn't know these on its own.
    let fileMenuItem = NSMenuItem()
    main.addItem(fileMenuItem)
    let fileMenu = NSMenu(title: "File")
    fileMenuItem.submenu = fileMenu
    fileMenu.delegate = delegate
    let trainerStatus = NSMenuItem(title: "NAM Trainer: nessun indirizzo impostato", action: nil, keyEquivalent: "")
    trainerStatus.isEnabled = false
    fileMenu.addItem(trainerStatus)
    delegate.trainerStatusItem = trainerStatus
    fileMenu.addItem(NSMenuItem.separator())
    let setHostItem = NSMenuItem(title: "Indirizzo del trainer…", action: #selector(AppDelegate.setTrainerHost(_:)), keyEquivalent: "")
    setHostItem.target = delegate
    fileMenu.addItem(setHostItem)
    let browseItem = NSMenuItem(title: "Sfoglia catture da esportare…", action: #selector(AppDelegate.browseCapturesFolder(_:)), keyEquivalent: "")
    browseItem.target = delegate
    fileMenu.addItem(browseItem)

    let editMenuItem = NSMenuItem()
    main.addItem(editMenuItem)
    let editMenu = NSMenu(title: "Modifica")
    editMenuItem.submenu = editMenu
    editMenu.addItem(withTitle: "Annulla", action: Selector(("undo:")), keyEquivalent: "z")
    editMenu.addItem(withTitle: "Ripeti", action: Selector(("redo:")), keyEquivalent: "Z")
    editMenu.addItem(NSMenuItem.separator())
    editMenu.addItem(withTitle: "Taglia", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
    editMenu.addItem(withTitle: "Copia", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
    editMenu.addItem(withTitle: "Incolla", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
    editMenu.addItem(withTitle: "Seleziona tutto", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")

    let windowMenuItem = NSMenuItem()
    main.addItem(windowMenuItem)
    let windowMenu = NSMenu(title: "Finestra")
    windowMenuItem.submenu = windowMenu
    windowMenu.addItem(withTitle: "Minimizza", action: #selector(NSWindow.miniaturize(_:)), keyEquivalent: "m")
    windowMenu.addItem(withTitle: "Chiudi finestra", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
    NSApp.windowsMenu = windowMenu

    return main
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.mainMenu = buildMainMenu(delegate)
app.run()
