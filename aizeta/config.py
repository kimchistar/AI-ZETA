import os

DATA_ROOT = os.environ.get("AIZETA_DATA", "data")
RUNS = ["2_TRIP1", "8_TRIP2", "31_A003", "26_V19_5A", "21_V19_1A"]
LAUE_DIRS = {c: os.path.join(DATA_ROOT, "laue", c) for c in RUNS}
TEMP_FILES = {c: os.path.join(DATA_ROOT, "temperature", f"{c}.xlsx") for c in RUNS}
