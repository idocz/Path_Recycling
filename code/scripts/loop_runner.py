import sys
from os.path import join
import os


scene = "jplext"
scene = "smoke"
# scene = "smallcf"
# renderer = "rr"
renderer = "seed"
script_name = f"{renderer}_inverse_{scene}_oneStage_loop.py"
script_path = join("scripts", script_name)
# Nrs =      [10]
# Nrs =      [1, 2, 5, 10 ,20, 30]
# Nrs =      [1, 2, 5, 20 ,30]
Nrs =      [10]
# Nrs =      [5, 10, 20 ,30]
# runtimes = [240] * len(Nrs)
runtimes = [4800] * len(Nrs)
print("Nrs=",Nrs)
print("rumtimes=",runtimes)

for to_sort in [0]:
    for Nr, runtime in zip(Nrs, runtimes):
        # if Nr  == 1  or (Nr == 10 and to_sort == True) or (Nr == 2 and to_sort == True) or (Nr == 5 and to_sort == True):
        #     continue

        print("###############################")
        print("###############################")
        print(f"############Nr={Nr}#################")
        print("###############################")
        print("###############################")
        print("to_sort=", to_sort)
        try:
            script_command = f"python {script_path} {Nr} {runtime} {to_sort}"
            print("running:", script_command)
            os.system(script_command)
        except KeyboardInterrupt:
            continue