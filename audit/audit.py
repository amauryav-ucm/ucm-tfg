import psutil
import subprocess
import time
import os
import json
import sys

init_pids = set(psutil.pids())
instance = subprocess.Popen(sys.argv[1:])

t = 0
step = 0.1
processes = {}

start_time = time.time()

while time.time() - start_time < 10:
    curr_pids = set(psutil.pids())
    new_pids = curr_pids - init_pids
    for pid in new_pids:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                continue

            snapshot = {
                "rss": p.memory_info().rss,
                "cpu": p.cpu_percent(interval=0.1) / psutil.cpu_count(),
                "status": p.status(),
                "cmd": " ".join(p.cmdline()),
            }

            processes.setdefault(pid, {})[t] = snapshot

        except psutil.NoSuchProcess:
            pass

    time.sleep(step)
    t += step

with open("data.json", "w") as f:
    json.dump(processes, f, indent=4)
