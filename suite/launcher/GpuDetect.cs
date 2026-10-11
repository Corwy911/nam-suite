// Which PyTorch build suits this PC? Run by the NAM Suite installer before its "engine" page:  gpu-detect.exe <it|en> <output file>
// Writes lines "key=value": variant (nvidia|amd|cpu), gpu, cpu, reason. Same rules as the old installer (Stub.cs, NAM Trainer 1.0):
//   NVIDIA wins over AMD; AMD only for Radeons worth trying with ROCm (RX / PRO / Ryzen AI iGPUs) and only on Windows 11; else CPU.
// Test hook: NAM_FAKE_ADAPTERS="10DE|NVIDIA GeForce RTX 4070;1002|AMD Radeon RX 7900 XTX" replaces the WMI query.
// Targets C# 5 / .NET Framework 4.x.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Management;
using System.Text;
using System.Text.RegularExpressions;

static class GpuDetect
{
    static string lang = "en";
    static string L(string it, string en) { return lang == "it" ? it : en; }

    sealed class Adapter { public string Name, Vendor; }

    static List<Adapter> Detect()
    {
        List<Adapter> list = new List<Adapter>();
        string fake = Environment.GetEnvironmentVariable("NAM_FAKE_ADAPTERS");
        if (fake != null)
        {
            foreach (string item in fake.Split(';'))
            {
                string[] p = item.Split(new char[] { '|' }, 2);
                if (p.Length == 2) { Adapter a = new Adapter(); a.Vendor = p[0].Trim().ToUpperInvariant(); a.Name = p[1].Trim(); list.Add(a); }
            }
            return list;
        }
        try
        {
            using (ManagementObjectSearcher s = new ManagementObjectSearcher("SELECT Name, PNPDeviceID FROM Win32_VideoController"))
            using (ManagementObjectCollection r = s.Get())
                foreach (ManagementBaseObject o in r)
                {
                    Match m = Regex.Match(Convert.ToString(o["PNPDeviceID"]) ?? "", "VEN_([0-9A-Fa-f]{4})");
                    Adapter a = new Adapter();
                    a.Name = (Convert.ToString(o["Name"]) ?? "").Trim();
                    a.Vendor = m.Success ? m.Groups[1].Value.ToUpperInvariant() : "";
                    list.Add(a);
                }
        }
        catch (Exception) { }
        return list;
    }

    static string Cpu()
    {
        try
        {
            using (ManagementObjectSearcher s = new ManagementObjectSearcher("SELECT Name FROM Win32_Processor"))
            using (ManagementObjectCollection r = s.Get())
                foreach (ManagementBaseObject o in r) return Regex.Replace((Convert.ToString(o["Name"]) ?? "").Trim(), "\\s+", " ");
        }
        catch (Exception) { }
        return "";
    }

    static bool AmdOfficiallySupported(string name)
    {
        return Regex.IsMatch(name, @"RX\s*(9\d{3}|7[789]\d{2})\b", RegexOptions.IgnoreCase)
            || Regex.IsMatch(name, @"PRO\s*(W7[789]\d{2}|R9\d{3})\b", RegexOptions.IgnoreCase)
            || Regex.IsMatch(name, @"\b(8050S|8060S|890M|880M)\b", RegexOptions.IgnoreCase);
    }

    static bool AmdCandidate(string name)
    {
        return Regex.IsMatch(name, @"Radeon.*\b(RX|PRO|AI PRO)\b", RegexOptions.IgnoreCase) || AmdOfficiallySupported(name);
    }

    static int Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "it") lang = "it";
        string outFile = args.Length > 1 ? args[1] : null;
        List<Adapter> adapters = Detect();
        string variant = "cpu", gpu = "", reason = "";
        foreach (Adapter a in adapters)
            if (a.Vendor == "10DE") { variant = "nvidia"; gpu = a.Name; reason = L("scheda NVIDIA rilevata", "NVIDIA card detected"); break; }
        if (variant == "cpu")
        {
            string amdOther = "";
            foreach (Adapter a in adapters)
            {
                if (a.Vendor != "1002") continue;
                if (AmdCandidate(a.Name))
                {
                    if (Environment.OSVersion.Version.Build < 22000) { amdOther = a.Name + L(" (l'accelerazione AMD richiede Windows 11)", " (AMD acceleration needs Windows 11)"); continue; }
                    variant = "amd"; gpu = a.Name;
                    reason = AmdOfficiallySupported(a.Name)
                        ? L("scheda AMD supportata dall'accelerazione ROCm", "AMD card supported by ROCm acceleration")
                        : L("scheda AMD non nell'elenco ufficiale ROCm: provo comunque; se non è utilizzabile si ripiega sulla CPU",
                            "AMD card not on the official ROCm list: trying anyway; if it is not usable the CPU version is used");
                    break;
                }
                amdOther = a.Name;
            }
            if (variant == "cpu")
                reason = amdOther.Length > 0
                    ? L("scheda AMD non adatta all'accelerazione ROCm (", "AMD card not suited to ROCm acceleration (") + amdOther + L("): uso la CPU", "): using the CPU")
                    : L("nessuna scheda NVIDIA o AMD adatta: uso la CPU", "no suitable NVIDIA or AMD card: using the CPU");
        }
        string text = "variant=" + variant + "\r\ngpu=" + gpu + "\r\ncpu=" + Cpu() + "\r\nreason=" + reason + "\r\n";
        if (outFile != null) File.WriteAllText(outFile, text, new UTF8Encoding(false));
        else Console.Write(text);
        return 0;
    }
}
