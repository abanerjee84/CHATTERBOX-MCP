"""Start the shared service outside an MCP client's kill-on-close process job."""

import subprocess


def process_running(pid):
    import win32api
    import win32con
    import win32process

    try:
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
    except Exception as exc:
        # Access denied cannot establish that a process has exited.
        return getattr(exc, "winerror", None) == 5
    try:
        return win32process.GetExitCodeProcess(handle) == 259
    finally:
        handle.Close()


def start_service(command, env, cwd):
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    services = startup = process = params = result = None
    try:
        services = win32com.client.GetObject(
            r"winmgmts:{impersonationLevel=impersonate}!\\.\root\cimv2"
        )
        startup = services.Get("Win32_ProcessStartup").SpawnInstance_()
        startup.ShowWindow = 0
        startup.CreateFlags = (
            subprocess.CREATE_NO_WINDOW | subprocess.CREATE_BREAKAWAY_FROM_JOB
        )
        startup.EnvironmentVariables = [f"{key}={value}" for key, value in env.items()]
        process = services.Get("Win32_Process")
        params = process.Methods_("Create").InParameters.SpawnInstance_()
        params.CommandLine = subprocess.list2cmdline(command)
        params.CurrentDirectory = str(cwd)
        params.ProcessStartupInformation = startup
        result = process.ExecMethod_("Create", params)
        if result.ReturnValue:
            raise RuntimeError(
                f"Windows service launch failed with WMI code {result.ReturnValue}"
            )
        return int(result.ProcessId)
    finally:
        result = params = process = startup = services = None
        pythoncom.CoUninitialize()
