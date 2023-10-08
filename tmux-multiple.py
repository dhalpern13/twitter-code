from itertools import product
from math import ceil
from os import system
panes_per_window = 9
instances = 42
windows = ceil(instances / panes_per_window)

for window in range(windows):
    for _ in range(panes_per_window - 1):
        system('tmux split-window')
        system('tmux select-layout tiled')
    if window < windows - 1:
        system('tmux new-window')
for instance in range(instances):
    window = instance // panes_per_window
    pane = instance % panes_per_window
    system(f'tmux send-keys -t {window}.{pane} "python3 new_run_one.py {instance}" Enter')