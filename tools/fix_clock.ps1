# Sets the Windows clock from an NTP server to the millisecond (Set-Date in PowerShell 5.1 is not
# reliable below one second). Run in an ADMIN PowerShell:
#   powershell -ExecutionPolicy Bypass -File tools\fix_clock.ps1
# Chert declines calls when this PC's clock is more than 0.25 s off.
param([string]$Server = 'time.google.com')

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class Clock {
    [StructLayout(LayoutKind.Sequential)]
    public struct SYSTEMTIME { public ushort Year, Month, DayOfWeek, Day, Hour, Minute, Second, Milliseconds; }
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool SetSystemTime(ref SYSTEMTIME st);
    public static bool Set(DateTime utc) {
        var st = new SYSTEMTIME { Year = (ushort)utc.Year, Month = (ushort)utc.Month, Day = (ushort)utc.Day,
            Hour = (ushort)utc.Hour, Minute = (ushort)utc.Minute, Second = (ushort)utc.Second,
            Milliseconds = (ushort)utc.Millisecond };
        return SetSystemTime(ref st);
    }
}
'@

# One SNTP exchange; returns this PC's offset from the server in seconds (positive = PC is behind).
function Get-NtpOffset([string]$server) {
    $req = New-Object byte[] 48; $req[0] = 0x1B
    $udp = New-Object Net.Sockets.UdpClient; $udp.Client.ReceiveTimeout = 3000
    try {
        $udp.Connect($server, 123)
        $t1 = [DateTime]::UtcNow
        [void]$udp.Send($req, 48)
        $ep = New-Object Net.IPEndPoint([Net.IPAddress]::Any, 0)
        $resp = $udp.Receive([ref]$ep)
        $t4 = [DateTime]::UtcNow
    } finally { $udp.Close() }
    function ts([byte[]]$b, [int]$i) {
        $sec = [uint64]0; $frac = [uint64]0
        for ($k = 0; $k -lt 4; $k++) { $sec = ($sec -shl 8) -bor $b[$i + $k]; $frac = ($frac -shl 8) -bor $b[$i + 4 + $k] }
        (New-Object DateTime(1900, 1, 1, 0, 0, 0, [DateTimeKind]::Utc)).AddSeconds($sec + $frac / 4294967296.0)
    }
    $t2 = ts $resp 32; $t3 = ts $resp 40
    (($t2 - $t1).TotalSeconds + ($t3 - $t4).TotalSeconds) / 2
}

function Measure-Offset { $o = @(1..5 | ForEach-Object { Get-NtpOffset $Server }) | Sort-Object; $o[2] }

$before = Measure-Offset
"Before: this PC is {0:N3} s {1} {2}" -f [Math]::Abs($before), $(if ($before -gt 0) { 'behind' } else { 'ahead of' }), $Server
if (-not [Clock]::Set([DateTime]::UtcNow.AddSeconds($before))) {
    throw "Could not set the clock (error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())). Run this in an ADMIN PowerShell."
}
Start-Sleep -Milliseconds 500
$after = Measure-Offset
"After:  this PC is {0:N3} s {1} {2}" -f [Math]::Abs($after), $(if ($after -gt 0) { 'behind' } else { 'ahead of' }), $Server
if ([Math]::Abs($after) -lt 0.05) { 'OK: within 50 ms. Chert will accept calls.' } else { 'Still off: run this script once more.' }
