"""Own hidden Windows child trees with a kill-on-close Job Object.

Children are assigned while suspended, before Python launchers can create their
real interpreter. No shell is used. --smoke launches only sleeping Python code.
"""

import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from uuid import uuid4

SIZE_T = ctypes.c_size_t
HANDLE = w.HANDLE
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
WAIT_OBJECT_0, WAIT_TIMEOUT = 0, 258
_spawn_lock = threading.Lock()


class SecurityAttributes(ctypes.Structure):
    _fields_ = [('nLength', w.DWORD), ('lpSecurityDescriptor', w.LPVOID), ('bInheritHandle', w.BOOL)]


class StartupInfo(ctypes.Structure):
    _fields_ = [('cb', w.DWORD), ('lpReserved', w.LPWSTR), ('lpDesktop', w.LPWSTR), ('lpTitle', w.LPWSTR)] + [
        (name, w.DWORD) for name in ('dwX', 'dwY', 'dwXSize', 'dwYSize', 'dwXCountChars', 'dwYCountChars', 'dwFillAttribute', 'dwFlags')
    ] + [('wShowWindow', w.WORD), ('cbReserved2', w.WORD), ('lpReserved2', ctypes.POINTER(w.BYTE)),
         ('hStdInput', HANDLE), ('hStdOutput', HANDLE), ('hStdError', HANDLE)]


class StartupInfoEx(ctypes.Structure):
    _fields_ = [('StartupInfo', StartupInfo), ('lpAttributeList', w.LPVOID)]


class ProcessInfo(ctypes.Structure):
    _fields_ = [('hProcess', HANDLE), ('hThread', HANDLE), ('dwProcessId', w.DWORD), ('dwThreadId', w.DWORD)]


class BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
               ('LimitFlags', w.DWORD), ('MinimumWorkingSetSize', SIZE_T), ('MaximumWorkingSetSize', SIZE_T),
               ('ActiveProcessLimit', w.DWORD), ('Affinity', SIZE_T), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                                                    'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IOCounters),
               ('ProcessMemoryLimit', SIZE_T), ('JobMemoryLimit', SIZE_T),
               ('PeakProcessMemoryUsed', SIZE_T), ('PeakJobMemoryUsed', SIZE_T)]


def _api():
    if os.name != 'nt':
        raise RuntimeError('WindowsJob requires Windows.')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    declarations = {
        'CreateJobObjectW': ([w.LPVOID, w.LPCWSTR], HANDLE),
        'SetInformationJobObject': ([HANDLE, ctypes.c_int, w.LPVOID, w.DWORD], w.BOOL),
        'AssignProcessToJobObject': ([HANDLE, HANDLE], w.BOOL),
        'CloseHandle': ([HANDLE], w.BOOL),
        'CreateFileW': ([w.LPCWSTR, w.DWORD, w.DWORD, ctypes.POINTER(SecurityAttributes), w.DWORD, w.DWORD, HANDLE], HANDLE),
        'CreateProcessW': ([w.LPCWSTR, w.LPWSTR, w.LPVOID, w.LPVOID, w.BOOL, w.DWORD, w.LPVOID, w.LPCWSTR,
                            ctypes.POINTER(StartupInfoEx), ctypes.POINTER(ProcessInfo)], w.BOOL),
        'InitializeProcThreadAttributeList': ([w.LPVOID, w.DWORD, w.DWORD, ctypes.POINTER(SIZE_T)], w.BOOL),
        'UpdateProcThreadAttribute': ([w.LPVOID, w.DWORD, SIZE_T, w.LPVOID, SIZE_T, w.LPVOID, w.LPVOID], w.BOOL),
        'DeleteProcThreadAttributeList': ([w.LPVOID], None),
        'ResumeThread': ([HANDLE], w.DWORD),
        'TerminateProcess': ([HANDLE, w.UINT], w.BOOL),
        'WaitForSingleObject': ([HANDLE, w.DWORD], w.DWORD),
        'GetExitCodeProcess': ([HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
        'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], HANDLE),
        'IsProcessInJob': ([HANDLE, HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
    }
    for name, (args, result) in declarations.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = args, result
    return kernel


def _check(success):
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())
    return success


class ManagedProcess:
    def __init__(self, kernel, handle, pid):
        self.kernel, self.handle, self.pid = kernel, handle, int(pid)
        self.exit_code = None

    def poll(self):
        if self.handle is None:
            return self.exit_code
        result = self.kernel.WaitForSingleObject(self.handle, 0)
        if result == WAIT_TIMEOUT:
            return None
        if result != WAIT_OBJECT_0:
            raise ctypes.WinError(ctypes.get_last_error())
        code = w.DWORD()
        _check(self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)))
        self.exit_code = int(code.value)
        return self.exit_code


class WindowsJob:
    def __init__(self):
        self.kernel = _api()
        self.handle = _check(self.kernel.CreateJobObjectW(None, None))
        self.processes = []
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            _check(self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise

    def __enter__(self):
        if self.handle is None:
            raise RuntimeError('Job is already closed.')
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def spawn(self, command, *, cwd, env, stdout, stderr):
        if self.handle is None:
            raise RuntimeError('Cannot spawn into a closed job.')
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) or '\0' in arg for arg in command):
            raise ValueError('Command must be a nonempty list of strings without NUL.')
        executable = Path(command[0]).resolve(strict=True)
        working_directory = str(Path(cwd).resolve(strict=True))
        if not isinstance(env, dict) or any(not isinstance(k, str) or not k or '=' in k or '\0' in k or
                                           not isinstance(v, str) or '\0' in v for k, v in env.items()):
            raise ValueError('Environment must contain valid string names and values.')
        environment = ctypes.create_unicode_buffer('\0'.join(f'{k}={v}' for k, v in sorted(env.items(), key=lambda kv: kv[0].upper())) + '\0\0')
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline([str(executable), *command[1:]]))
        paths = [Path(stdout).resolve(), Path(stderr).resolve()]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        kernel, handles, attributes = self.kernel, [], None
        info = ProcessInfo()
        created = adopted = False
        # Restrict inheritance to these three handles, even if another thread has
        # unrelated inheritable handles. The job handle itself is never inherited.
        with _spawn_lock:
            try:
                security = SecurityAttributes(ctypes.sizeof(SecurityAttributes), None, True)
                for path, access, disposition in [('NUL', 0x80000000, 3), (str(paths[0]), 0x40000000, 2), (str(paths[1]), 0x40000000, 2)]:
                    handle = kernel.CreateFileW(path, access, 3, ctypes.byref(security), disposition, 0x80, None)
                    if handle == INVALID_HANDLE_VALUE:
                        raise ctypes.WinError(ctypes.get_last_error())
                    handles.append(handle)
                size = SIZE_T()
                kernel.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
                if not size.value:
                    raise ctypes.WinError(ctypes.get_last_error())
                buffer = ctypes.create_string_buffer(size.value)
                pointer = ctypes.cast(buffer, w.LPVOID)
                _check(kernel.InitializeProcThreadAttributeList(pointer, 1, 0, ctypes.byref(size)))
                attributes = pointer
                inherited = (HANDLE * 3)(*handles)
                _check(kernel.UpdateProcThreadAttribute(attributes, 0, 0x20002, ctypes.cast(inherited, w.LPVOID),
                                                       ctypes.sizeof(inherited), None, None))
                startup = StartupInfoEx()
                startup.StartupInfo.cb = ctypes.sizeof(startup)
                startup.StartupInfo.dwFlags = 0x100  # STARTF_USESTDHANDLES
                startup.StartupInfo.hStdInput, startup.StartupInfo.hStdOutput, startup.StartupInfo.hStdError = handles
                startup.lpAttributeList = attributes
                flags = 0x4 | 0x08000000 | 0x400 | 0x80000  # suspended, no window, Unicode env, extended startup
                _check(kernel.CreateProcessW(str(executable), command_line, None, None, True, flags,
                                             ctypes.cast(environment, w.LPVOID), working_directory,
                                             ctypes.byref(startup), ctypes.byref(info)))
                created = True
                _check(kernel.AssignProcessToJobObject(self.handle, info.hProcess))
                if kernel.ResumeThread(info.hThread) == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
                process = ManagedProcess(kernel, info.hProcess, info.dwProcessId)
                self.processes.append(process)
                adopted = True
                return process
            finally:
                if created and not adopted:
                    kernel.TerminateProcess(info.hProcess, 1)
                    kernel.WaitForSingleObject(info.hProcess, 5000)
                    kernel.CloseHandle(info.hProcess)
                if info.hThread:
                    kernel.CloseHandle(info.hThread)
                if attributes:
                    kernel.DeleteProcThreadAttributeList(attributes)
                for handle in handles:
                    kernel.CloseHandle(handle)

    def close(self):
        if self.handle is None:
            return
        handle, self.handle = self.handle, None
        # Closing our sole, non-inheritable job handle kills every descendant.
        _check(self.kernel.CloseHandle(handle))
        errors = []
        for process in self.processes:
            try:
                if self.kernel.WaitForSingleObject(process.handle, 10000) != WAIT_OBJECT_0:
                    raise TimeoutError(f'Owned process {process.pid} did not stop within 10 seconds.')
                process.poll()
            except Exception as exc:
                errors.append(exc)
            finally:
                self.kernel.CloseHandle(process.handle)
                process.handle = None
        if errors:
            raise errors[0]


def smoke():
    """A bounded sleeping child/grandchild check; no model, server or network."""
    directory = Path(__file__).resolve().parents[2] / 'bench/results' / ('windows_job_smoke_' + uuid4().hex)
    directory.mkdir(parents=True)
    job = WindowsJob()
    descendant_handle = None
    report = {'kind': 'harmless_windows_process_tree_smoke'}
    try:
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        process = job.spawn([sys.executable, '-c', code], cwd=directory, env=os.environ.copy(),
                            stdout=directory / 'stdout.log', stderr=directory / 'stderr.log')
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            text = (directory / 'stdout.log').read_text().strip()
            if text:
                break
            if process.poll() is not None:
                raise RuntimeError('Smoke child exited before creating its descendant.')
            time.sleep(.05)
        else:
            raise TimeoutError('Smoke child startup timed out.')
        descendant_pid = int(text)
        descendant_handle = _check(job.kernel.OpenProcess(0x100000 | 0x1000, False, descendant_pid))
        member = w.BOOL()
        _check(job.kernel.IsProcessInJob(descendant_handle, job.handle, ctypes.byref(member)))
        if not member.value:
            raise RuntimeError('Descendant escaped the owned job.')
        report.update(pid=process.pid, descendant_pid=descendant_pid, descendant_in_job=True)
        job.close()
        if job.kernel.WaitForSingleObject(descendant_handle, 5000) != WAIT_OBJECT_0:
            raise RuntimeError('Descendant survived job close.')
        report.update(status='passed', parent_exit_code=process.poll(), descendant_stopped=True)
        job.close()  # idempotency
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            job.close()
        finally:
            if descendant_handle:
                job.kernel.CloseHandle(descendant_handle)
            (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return {'report': str(directory / 'report.json'), **report}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    options = parser.parse_args()
    if options.smoke:
        print(json.dumps(smoke(), indent=2))
    else:
        parser.print_help()
