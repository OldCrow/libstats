# Opts a process out of Windows power throttling (EcoQoS): PROCESS_POWER_THROTTLING_EXECUTION_SPEED
# in ControlMask with StateMask 0. Per process; no system setting changes.
Add-Type -Namespace Win32 -Name Throttle -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)]
public struct PROCESS_POWER_THROTTLING_STATE { public uint Version; public uint ControlMask; public uint StateMask; }
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool SetProcessInformation(IntPtr hProcess, int ProcessInformationClass,
    ref PROCESS_POWER_THROTTLING_STATE info, uint size);
'@
function Disable-PowerThrottling([System.Diagnostics.Process]$p) {
    $st = New-Object Win32.Throttle+PROCESS_POWER_THROTTLING_STATE
    $st.Version = 1; $st.ControlMask = 1; $st.StateMask = 0
    [Win32.Throttle]::SetProcessInformation($p.Handle, 4, [ref]$st, 12)  # 4 = ProcessPowerThrottling
}
Add-Type -Namespace Win32 -Name ThrottleQ -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)]
public struct PROCESS_POWER_THROTTLING_STATE { public uint Version; public uint ControlMask; public uint StateMask; }
[DllImport("kernel32.dll", SetLastError = true)]
public static extern bool GetProcessInformation(IntPtr hProcess, int ProcessInformationClass,
    ref PROCESS_POWER_THROTTLING_STATE info, uint size);
'@
function Get-PowerThrottling([System.Diagnostics.Process]$p) {
    $st = New-Object Win32.ThrottleQ+PROCESS_POWER_THROTTLING_STATE
    $st.Version = 1
    $ok = [Win32.ThrottleQ]::GetProcessInformation($p.Handle, 4, [ref]$st, 12)
    "ok=$ok control=$($st.ControlMask) state=$($st.StateMask)"
}
