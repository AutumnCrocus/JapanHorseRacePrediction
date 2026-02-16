import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

def init_worker():
    print(f"Worker {os.getpid()} started")

def job(x):
    return x * x

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("Main start")
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as ex:
        futures = {ex.submit(job, i): i for i in range(5)}
        for f in as_completed(futures):
            print(f.result())
    print("Main done")
