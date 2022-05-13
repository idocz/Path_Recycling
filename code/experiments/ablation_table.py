import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from utils import get_images_from_TB, animate, get_scalars_from_TB
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def get_zero_after_dot(num):
    temp_num = num
    count = 0
    while int(temp_num)==0:
        count += 1
        temp_num *= 10
    return count

exp_dict_jplext = {
    #[Nr,sort] : exp_name
    (1, True) : "2602-0128-26_jplext_Nr=1_ss=2.0e+09_tosort=1",
    # (1, False): "2602-1346-38_jplext_Nr=1_ss=2.0e+09_tosort=0", # NO SORT
    (1, False): "0603-0818-17_jplext_Nr=1_ss=2.0e+09_tosort=0", # NO SORT

    (2, True) : "2602-0331-31_jplext_Nr=2_ss=2.0e+09_tosort=1",
    (2, False): "2602-1549-37_jplext_Nr=2_ss=2.0e+09_tosort=0", # NO SORT

    (5, True) : "2602-0534-36_jplext_Nr=5_ss=2.0e+09_tosort=1",
    (5, False): "2602-1752-39_jplext_Nr=5_ss=2.0e+09_tosort=0", # NO SORT

    (10, True): "2602-0737-36_jplext_Nr=10_ss=2.0e+09_tosort=1",
    (10, False): "2602-1955-41_jplext_Nr=10_ss=2.0e+09_tosort=0",

    (20, True): "2602-0940-36_jplext_Nr=20_ss=2.0e+09_tosort=1",
    (20, False): "2602-2158-41_jplext_Nr=20_ss=2.0e+09_tosort=0", # NO SORT

    (30, True): "2602-1143-37_jplext_Nr=30_ss=2.0e+09_tosort=1",
    (30, False): "2702-0001-39_jplext_Nr=30_ss=2.0e+09_tosort=0", # NO SORT
}


exp_dict_smallcf = {
    #[Nr,sort] : exp_name
    (1, True) : "2702-0803-21_smallcf_Nr=1_ss=5.00e+09_tosort=1",
    (1, False): "2402-2057-33_smallcf_Nr=1_ss=5.00e+09",

    (2, True) : "2502-0400-56_smallcf_Nr=2_ss=5.00e+09",
    (2, False): "2802-0642-37_smallcf_Nr=2_ss=5.00e+09_tosort=0",

    (5, True) : "2702-1306-45_smallcf_Nr=5_ss=5.00e+09_tosort=1",
    (5, False): "2802-1145-57_smallcf_Nr=5_ss=5.00e+09_tosort=0",

    (10, True): "2402-1249-05_smallcf_Nr=10_ss=5.00e+09",
    # (10, False): "2802-1649-13_smallcf_Nr=10_ss=5.00e+09_tosort=0",
    (10, False): "0103-2214-15_smallcf_Nr=10_ss=5.00e+09_tosort=0",

    (20, True): "2702-2035-59_smallcf_Nr=20_ss=5.00e+09_tosort=1",
    (20, False): "2802-2152-31_smallcf_Nr=20_ss=5.00e+09_tosort=0",

    (30, True): "2802-0139-16_smallcf_Nr=30_ss=5.00e+09_tosort=1",
    (30, False): "0103-0255-54_smallcf_Nr=30_ss=5.00e+09_tosort=0",
}



if __name__ == "__main__":

    scalar_name = "relative_dist1"


    print("###### JPLEXT scene ######\n")

    long_running_time = 2
    table_string = "Solitude Cloud row: "
    Nr_string = "    1             2          5           10         20         30"
    for params, name in exp_dict_jplext.items():
        if name is None:
            continue
        exp_dir = join("checkpoints",name)
        scalar_list = get_scalars_from_TB(exp_dir, scalar_name)
        ref = scalar_list[0].wall_time
        time_axis = np.array([(scalar.wall_time - ref) / 60 for scalar in scalar_list])
        value_axis = np.array([scalar.value for scalar in scalar_list])
        final_eps = value_axis[time_axis<=long_running_time][-1]
        round_digit = get_zero_after_dot(final_eps)
        final_round = round(final_eps, max(round_digit, 2))
        final_time = time_axis[value_axis>=0.5][-1]
        print(f"Nr={params[0]},to_sort={params[1]}: {final_eps}")
        # table_string += str(final_round)
        table_string += str(int(final_time))
        if params[1] == False:
            table_string += " & "
            print()
        else:
            table_string += "/"

    table_string = table_string[:-2] + "\\\\"
    print(Nr_string)
    print(table_string)
    print("\n\n")

    print("###### SMALLCF scene ######\n")
    long_running_time = 5
    table_string = ""
    for params, name in exp_dict_smallcf.items():
        if name is None:
            continue
        exp_dir = join("checkpoints",name)
        scalar_list = get_scalars_from_TB(exp_dir, scalar_name)
        ref = scalar_list[0].wall_time
        time_axis = np.array([(scalar.wall_time - ref) / 3600 for scalar in scalar_list])
        value_axis = np.array([scalar.value for scalar in scalar_list])
        final_eps = value_axis[time_axis<=long_running_time][-1]
        print(f"Nr={params[0]},to_sort={params[1]}: {final_eps}")
        if params[1] == False:
            print()

    table_string = table_string[:-2] + "\\\\"
    print(Nr_string)
    print(table_string)
    print("\n\n")