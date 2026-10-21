"""Later hardware/backend observation; does not score or select against test."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='4'
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import sys
import time
import winreg


class MemoryStatus(ctypes.Structure):
    _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD),
              *[(n,ctypes.c_ulonglong) for n in ('total_physical','available_physical','total_page','available_page','total_virtual','available_virtual','extended')]]


class ThreadEntry(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('usage',wintypes.DWORD),('tid',wintypes.DWORD),('pid',wintypes.DWORD),
              ('base',wintypes.LONG),('delta',wintypes.LONG),('flags',wintypes.DWORD)]


def thread_count(pid):
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    k.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD]
    k.Thread32First.argtypes=[wintypes.HANDLE,ctypes.POINTER(ThreadEntry)]
    k.Thread32Next.argtypes=[wintypes.HANDLE,ctypes.POINTER(ThreadEntry)]
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    h=k.CreateToolhelp32Snapshot(4,0)
    if h==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    entry=ThreadEntry()
    entry.size=ctypes.sizeof(entry)
    count=0
    try:
        ok=k.Thread32First(h,ctypes.byref(entry))
        while ok:
            count+=int(entry.pid==pid)
            ok=k.Thread32Next(h,ctypes.byref(entry))
    finally:
        k.CloseHandle(h)
    return count


def main():
    import numpy as np
    from threadpoolctl import threadpool_info
    from common import load_inputs
    from submission import predict
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    memory=MemoryStatus()
    memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    bank,arrays=load_inputs('validation')
    result=predict(bank,arrays,Path('runs/model_bundle'))
    backend=threadpool_info()
    observation={'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'cpu':cpu,'ram_physical_bytes':memory.total_physical,'ram_gib':memory.total_physical/2**30,
        'backend_info':backend,'thread_count_after_validation_prediction':thread_count(os.getpid()),
        'logical_cpus':os.cpu_count(),'context':'later validation-only prediction process; no test score rerun',
        'thread_interpretation':'Loaded backend caps and sampled process thread count; not historical monitoring of original final test.',
        'predictions':len(result)}
    Path('runs/supplement').mkdir(exist_ok=True)
    Path('runs/supplement/hardware.json').write_text(json.dumps(observation,indent=2))
    old=Path('runs/reproduction.json')
    backup=Path('runs/reproduction-original.json')
    if not backup.exists():
        backup.write_bytes(old.read_bytes())
    repro=json.loads(old.read_text())
    repro['environment']['cpu']=cpu
    repro['environment']['ram_physical_bytes']=memory.total_physical
    repro['environment']['hardware_observation_utc']=observation['observed_utc']
    repro['environment']['hardware_observation_source']='runs/supplement/hardware.json (later observation)'
    old.write_text(json.dumps(repro,indent=2))
    print(json.dumps(observation,indent=2))


if __name__=='__main__':
    main()
