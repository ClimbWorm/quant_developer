# -*- coding: utf-8 -*-

import os, sys
import importlib
import pprint



def load_param_list(seed_locator):
    task_path = os.path.dirname(__file__)
    task_path = os.path.join(task_path, "../")
    sys.path.append(task_path)

    para_list_lib = importlib.import_module(seed_locator)

    return para_list_lib.param_list

def for_recursive(loop_n, cur_idx_list, last_idx, cur_idx_ptr, seed_list, param_table, process_func):
    
    if (cur_idx_ptr == last_idx):
        for i in range(0, len(seed_list[cur_idx_ptr])):
            cur_idx_list[cur_idx_ptr] = i
            # print(cur_idx_list)
            process_func(seed_list, cur_idx_list, param_table)
    else:
        for i in range(0, len(seed_list[cur_idx_ptr])):
            cur_idx_list[cur_idx_ptr] = i
            for_recursive(loop_n-1, cur_idx_list, last_idx, cur_idx_ptr+1, seed_list, param_table, process_func)
    
def collect_line(seed_list, idx_list, param_table):
    line_list = list()

    for i in range(0, len(idx_list)):
        idx = idx_list[i]
        line_list.append(seed_list[i][idx])
    
    param_table.append(line_list)


def generate_param_list(seed_locator, output_file_name):
    param_list = load_param_list(seed_locator)

    param_len = len(param_list)
    print("==========> Generating parameter list!")
    pprint.pprint(param_list)
    header_list = list()
    param_table = list()
    seed_list = list()
    for i in range(0, len(param_list)):
        param = param_list[i]
        
        header_list.append((param["name"], param["type"]))
        seed_list.append(param["values"])
    
    idx_list = [0]*param_len

    for_recursive(param_len, idx_list, param_len-1, 0, seed_list, param_table, collect_line)    
    
    # print(param_dict)
    # print(header_list)
    # print(param_len)
    # print(seed_list)
    # print(param_table)

    with open(output_file_name, 'w+') as of:

        line_list = ["id"]
        for head_name, type in header_list:
            line_list.append(head_name)
        
        line_str = ','.join(line_list)
        line_str += '\n'

        of.writelines([line_str])

        id = 0
        for param in param_table:
            line_list = list()

            id_str = str(id)
            line_list.append(id_str)
            
            for i in range(0, param_len):
                dtype = header_list[i][1]

                if (dtype == "float"):
                    line_list.append("%f" % param[i])
                elif (dtype == "int"):
                    line_list.append("%d" % param[i])
                else:
                    line_list.append(str(param[i]))
            
            line_str = ','.join(line_list)
            line_str += '\n'
            of.writelines([line_str])

            id += 1
            


if __name__ == '__main__':
    task_work_path = "./cta_py/src/task/TestTask/work"
    param_file = os.path.join(task_work_path, "param_list.csv")
    generate_param_list("task.TestTask.param_seed", param_file)
