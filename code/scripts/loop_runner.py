import sys
from os.path import join
import os


# scene = "jplext"
# renderer = "multi"
scene = "smallcf"
renderer = "rr"
script_name = f"{renderer}_inverse_{scene}_oneStage_loop.py"
script_path = join("scripts", script_name)
# Nrs =      [10]
Nrs =      [10]
# runtimes = [120, 1,1,1 ,1 ,1]
runtimes = [1800]*len(Nrs)
to_sort = 1
print("Nrs=",Nrs)
print("rumtimes=",runtimes)
print("to_sort=",to_sort)
for Nr, runtime in zip(Nrs, runtimes):
    print("###############################")
    print("###############################")
    print(f"############Nr={Nr}#################")
    print("###############################")
    print("###############################")
    try:
        script_command = f"python {script_path} {Nr} {runtime} {to_sort}"
        print("running:", script_command)
        os.system(script_command)
    except KeyboardInterrupt:
        continue