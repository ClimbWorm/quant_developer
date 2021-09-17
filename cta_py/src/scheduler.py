# -*- coding: utf-8 -*-
import os, sys
import datetime, time
# import util.config_reader
sche_path = os.path.dirname(__file__)
sys.path.append(sche_path)
from util import config_reader
from util import para_generator
from util import remote_proxy

def do_execute(rmt_proxy, rmt_work_dir, server, pyexe, params_lst):
    rmt_proxy.execute(rmt_work_dir, server, pyexe, params_lst)



class TaskScheduler:
    config = None
    # Todo 修改路径
    config_path = r"F:\pythonHistoricalTesting\pythonHistoricalTesting\cta_py\config"  # 懒得写到配置里了，配置到本地环境即可
    task_path = r"F:\pythonHistoricalTesting\pythonHistoricalTesting\cta_py\src\task"


    selected_task_id = -1
    batch_id = ""
    local_work_dir = ""
    working_param_file = ""
    working_param_block_list = []
    rmt_work_dir_list = [] #存放touch的batch_id文件 工作目录 命名类似于TestTask+时间戳（batch id）的命名格式

    def __init__(self) -> None:
        self.config = config_reader.ScheduleConfigReader(self.config_path)
        # self.result_address_list_on_servers()

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.batch_id = ts

    def select_task(self):
        task_id = 0
        print('-' * 50)
        for task_config in self.config.get_tasks():
            print(task_id, task_config['task_name'])
            task_id += 1
        print('-' * 50)

        selected_id = input('Select task:')
        selected_id = int(selected_id)
        if (selected_id >= 0 and selected_id < len(self.config.get_tasks())):
            self.selected_task_id = selected_id
        else:
            raise Exception("Bad input leads to exception")

    def generate_param_list(self):
        task_to_execute = self.config.get_tasks()[self.selected_task_id]
        task_name = task_to_execute['task_name']
        task_work_path = os.path.join(self.task_path, task_name, "work")
        print(task_work_path)

        if (not os.path.exists(task_work_path)):
            os.mkdir(task_work_path)

        task_prefix = task_to_execute['task_local_path']
        batch_dir = os.path.join(task_work_path, task_prefix + self.batch_id)
        os.mkdir(batch_dir)
        self.local_work_dir = batch_dir

        param_seed = task_to_execute['param_seed']
        param_file = os.path.join(self.local_work_dir, "param_list.csv")

        para_generator.generate_param_list(param_seed, param_file)
        self.working_param_file = param_file

    def split_param_file_by_servers(self):
        server_list = self.config.get_servers()
        print(server_list)

        server_weight = list()
        for server in server_list:
            weight = server['core_count'] * server['core_weight']
            server_weight.append(weight)
            print('server:%s, weight: %d' % (server['server_name'], weight))

        total_weight = sum(server_weight)

        param_header = str()
        param_line_list = list()
        with open(self.working_param_file, 'r') as pf:
            param_header = pf.readline()

            param_line = pf.readline()
            while (param_line):
                param_line_list.append(param_line)

                param_line = pf.readline()

        print(param_header)
        # print(param_line_list)

        self.working_param_block_list.clear()

        total_lines = len(param_line_list)
        for i in range(0, len(server_list)):
            server = server_list[i]
            block_len = total_lines * server_weight[i] / total_weight
            block_len = int(block_len)
            print('=====> block lines count:' + str(block_len))
            print()

            if (i == len(server_list) - 1):
                block_param_list = param_line_list
            else:
                block_param_list = param_line_list[0:block_len]
            param_line_list = param_line_list[block_len:]

            server_block_file = server['server_name'] + '.csv'

            block_file_with_path = os.path.join(self.local_work_dir, server_block_file)
            with open(block_file_with_path, 'w+') as bf:
                bf.write(param_header)
                for line in block_param_list:
                    bf.write(line)

            self.working_param_block_list.append(server_block_file)

        # if (len(param_line_list) > 0):
        #     raise Exception('Remaining lines!')

        print(self.working_param_block_list)


    # def result_address_list_on_servers(self):
    #     servers_list = self.config.get_servers()
    #     for i in range(len(servers_list)):
    #         server = servers_list[i]
    #         self.server_result_path_list.append(os.path.join(server["work_path"], "result").replace('\\', '/'))
    #     print(self.server_result_path_list)

    # 分配并执行
    def dispatch_to_servers(self):
        # 把本地文件上传到服务器
        servers_list = self.config.get_servers()
        # pool = multiprocessing.Pool(len(servers_list))
        rmt_proxy = [None for i in range(len(servers_list))]
        for server_id in range(len(servers_list)):
            server = servers_list[server_id]
            hostname = server["ip"]
            port = server["port"]
            username = server["username"]
            password = server["password"]
            rmt_proxy[server_id] = remote_proxy.RemoteProxy(hostname, port, username, password)
            task_id = self.selected_task_id
            task_config = self.config.get_tasks()[task_id]

            file_list = list()
            file_list.append(task_config['py_location'])
            
            param_file = os.path.join(self.local_work_dir, self.working_param_block_list[server_id])
            file_list.append(param_file)
            file_list += task_config["refer_files"]
            # rmt_proxy[server_id].clear_work_dir(server['work_path'])
            rmt_work_dir = os.path.join(server['work_path'], task_config['task_name'] + self.batch_id).replace('\\', '/')
            # Todo
            self.rmt_work_dir_list.append(rmt_work_dir) # 因为每一个server对应的rmt_work_dir_list都不一样
            rmt_proxy[server_id].create_dir(rmt_work_dir)
            rmt_proxy[server_id].upload(file_list, rmt_work_dir)   # 把策略文件传到服务器的工作目录下

            # pyexe = os.path.split(task_config['py_location'])[-1]
            # pyexe = os.path.join(server['work_path'], pyexe).replace('\\', '/')
            pyexe = os.path.basename(task_config['py_location'])

            ret = rmt_proxy[server_id].execute(rmt_work_dir, server, pyexe,
                                    [
                                        str(self.batch_id),
                                        str(server['core_count']),
                                        str(self.working_param_block_list[server_id]),
                                        rmt_work_dir
                                    ])
            print("===================working_param_block_list:",self.working_param_block_list[server_id])
            print(rmt_work_dir)


            if ret:
                print('===========> Dispatch OK!')
            else:
                print('===========> Error while dispatching.')

        while self.keep_watching():
            print('Keep watching process...')

        for rmt_hdl in rmt_proxy:
            rmt_hdl.close()
        # 有时间把启动python策略任务面向对象化，先build执行器对象，把参数给这个对象，由对象调用策略执行，这样参数比较方便增加和管理

    def keep_watching(self):
        time.sleep(15)

        servers_list = self.config.get_servers()
        for server_id in range(len(servers_list)):
            rmt_work_dir = self.rmt_work_dir_list[server_id]
            server = servers_list[server_id]
            hostname = server["ip"]
            port = server["port"]
            username = server["username"]
            password = server["password"]
            rmt_proxy = remote_proxy.RemoteProxy(hostname, port, username, password)
            if not rmt_proxy.test_tag(rmt_work_dir, self.batch_id):
                rmt_proxy.close()
                return True
            else:
                print("{} successfully executed!".format(hostname))
            rmt_proxy.close()
        
        return False


    def collect_result(self):
        rmt_proxy = [None for i in range(len(self.rmt_work_dir_list))]
        for server_id in range(len(self.rmt_work_dir_list)):
            rmt_result_path = self.rmt_work_dir_list[server_id] + "/result/"
            local_result_path = self.local_work_dir + "/result"
            if not os.path.exists(local_result_path):
                os.makedirs(local_result_path)
            # print(os.walk(rmt_result_path))
            # files = os.listdir(rmt_result_path)  # file就为生成的所有结果文件

            server = self.config.get_servers()[server_id]
            hostname = server["ip"]
            port = server["port"]
            username = server["username"]
            password = server["password"]

            rmt_proxy[server_id] = remote_proxy.RemoteProxy(hostname, port, username, password)
            files = rmt_proxy[server_id].get_listdir(rmt_result_path)
            print(files)
            rmt_proxy[server_id].download(files, local_result_path)
            rmt_proxy[server_id].close()


if __name__ == '__main__':
    task_scheduler = TaskScheduler()
    task_scheduler.select_task()
    task_scheduler.generate_param_list()
    task_scheduler.split_param_file_by_servers()

    task_scheduler.dispatch_to_servers()


    task_scheduler.collect_result()
