# -*- coding: utf-8 -*-

import os
import json
import pprint


class ScheduleConfigReader:

    config_path = ""

    servers_config = dict()
    tasks_config = dict()


    def __init__(self, config_path) -> None:
        if (not os.path.exists(config_path)):
            raise Exception("Unknown config path!")
        
        server_config_file = os.path.join(config_path, "servers.json")
        print('Reading servers config from: ' + server_config_file)
        with open(server_config_file, "r") as cfg_f:
            self.servers_config = json.loads(cfg_f.read())
            pprint.pprint(self.servers_config)
        
        task_config_file = os.path.join(config_path, "tasks.json")
        print('Reading tasks config from: ' + task_config_file)
        with open(task_config_file, "r") as cfg_f:
            self.tasks_config = json.loads(cfg_f.read())  #将str类型转换成dict
            pprint.pprint(self.tasks_config)
    
    def get_servers(self):
        return self.servers_config['services']
    
    def get_tasks(self):
        return self.tasks_config['tasks']
    




if __name__ == '__main__':
    ScheduleConfigReader(r"F:\pythonHistoricalTesting\pythonHistoricalTesting\cta_py\config")