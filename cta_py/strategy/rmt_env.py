# -*- coding: utf-8 -*-

import os
import pandas as pd

class RemoteEnv:

    def __init__(self, argv:list):
        self.is_rmt = False

        if len(argv) == 1:
            return

        self.is_rmt = True
        print("remote processing started!")

        self.batch_id = argv[1]
        self.core_count = int(argv[2])
        self.param_file_name = argv[3]
        self.work_dir = argv[4]
        self.save_result_path = None

        if len(argv) > 5:
            self.add_params = argv[5:]
        else:
            self.add_params = None

    def is_remote(self):
        return self.is_rmt

    def prepare(self):
        rmt_work_dir = self.get_work_dir()
        self.save_result_path = os.path.join(rmt_work_dir, "result")
        if not os.path.exists(self.save_result_path):
            os.makedirs(self.save_result_path)


    def get_save_result_path(self):
        return self.save_result_path

    def get_batch_id(self):
        return self.batch_id

    def get_param_file(self):
        param_file_fullpath = os.path.join(self.work_dir, self.param_file_name)
        if not os.path.exists(param_file_fullpath):
            raise Exception('Can not find parameter file %s!' % param_file_fullpath)

        return param_file_fullpath
    
    def param_generator(self):
        df_param = pd.read_csv(self.get_param_file())
        for i in range(len(df_param)):
            line = df_param.iloc[i]
            yield line


    def get_cores(self):
        return self.core_count

    def get_work_dir(self):
        return self.work_dir

    def get_result_dir(self):
        return os.path.join(self.work_dir, 'result')

    def mark_end_with_success(self):
        if self.is_rmt:
            os.system("touch " + self.batch_id)
            print('Remote run ended successfully')

