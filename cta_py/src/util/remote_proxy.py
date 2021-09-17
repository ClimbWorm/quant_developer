import paramiko
import os
import pprint


class RemoteProxy(object):
    ssh_client = None
    ftp_client = None

    OS = 'linux'

    def __init__(self, hostname, port, username, password):
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_client.connect(hostname=hostname, port=port, username=username, password=password)
        self.ftp_client = self.ssh_client.open_sftp()
        self.hostname = hostname

    def set_os(self, OS):
        if (OS == 'linux'):
            self.OS = 'linux'
        elif (OS == 'windows'):
            self.OS = 'windows'

    def close(self):
        self.ftp_client.close()
        self.ssh_client.close()
    
    def _exe_with_check(self, cmd):
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        if len(stderr.read()) == 0:
            return True
        return False

    def clear_work_dir(self, dest: str):
        # Todo
        cmd = "rm %s* -rf" % dest
        print(cmd)
        # self.ssh_client.exec_command()

    def upload(self, filelist: list, dest: str):
        for file in filelist:  # 本地路径下的文件列表
            dest_file = os.path.join(dest, os.path.split(file)[-1]).replace('\\', '/')
            # print(os.path.exists(file))
            self.ftp_client.put(file, dest_file)
    def get_listdir(self,file_path):
        cmd = "cd " + file_path + ";ls"
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        ret = stdout.read()
        ret = str(ret)
        ret = ret.replace("b'", '')
        file_list_no_path = ret.split("\\n")[:-1]
        if len(stderr.read()) == 0:
            file_list_with_path = []
            for file in file_list_no_path:
                file_list_with_path.append(file_path + file)
            return file_list_with_path
        raise Exception("Something wrong happened when get listdir!")


    def download(self, filelist: list, local_dest: str):
        for file in filelist:  # 服务器路径下的文件列表
            dest_file = os.path.join(local_dest,os.path.split(file)[-1])
            self.ftp_client.get(file, dest_file)


    def create_result_dir(self,server_path,batch_id):
        # print(os.getcwd())
        # cmd = "mkdir " + str(server_path) + '/' + str(batch_id)
        cmd = "mkdir Iron_Musk/cta_work/result/" + str(batch_id)
        # print(str(server_path))
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        if len(stderr.read()) == 0:
            return True
        return False
    
    def create_dir(self, path):
        cmd = "mkdir -p %s" % path
        return self._exe_with_check(cmd)
    
    def _prepare_run_box(self, box_path, server_config):
        dir_soft_links = server_config['dir_soft_links']
        if not dir_soft_links:
            raise Exception('Unknown soft links!')
        
        cmd = 'cd ' + box_path
        self._exe_with_check(cmd)

        for soft_link in dir_soft_links:
            node_name = os.path.basename(soft_link)
            cmd = 'ln -s ' + soft_link + ' ./' + node_name
            print(cmd)
            self._exe_with_check(cmd)
    
    def execute(self, run_box_path:str, server_config:dict, pystrategy:str, args_list:list):
        # self._prepare_run_box(run_box_path, server_config)

        pre_cmd_list = ['cd ' + run_box_path]

        dir_soft_links = server_config['dir_soft_links']
        if not dir_soft_links:
            raise Exception('Unknown soft links!')
        
        for soft_link in dir_soft_links:
            node_name = os.path.basename(soft_link)
            cmd = 'ln -s ' + soft_link + ' ./' + node_name
            pre_cmd_list.append(cmd)
        
        pre_cmd = '; '.join(pre_cmd_list)

        cmd = '%s %s' % (server_config['engine'] ,pystrategy)
        for arg in args_list:
            cmd += (" " + arg)
        print("Executing at remote %s" % self.hostname)

        cmd = '; '.join([pre_cmd, cmd])
        # pprint.pprint(cmd)
        print(cmd)

        sh_cmd = 'cd ' + run_box_path + '; echo "%s" >> cmd.sh; chmod +x cmd.sh;' % cmd
        print(sh_cmd)
        stdin, stdout, stderr = self.ssh_client.exec_command(sh_cmd)

        print(stdout.read())

        cmd = 'source %s' % (run_box_path + '/cmd.sh >> %s/output.log 2>> %s/err.log' % (run_box_path, run_box_path))
        # cmd = 'cd ' + run_box_path + '; source cmd.sh;'
        print(cmd)
        channel = self.ssh_client.get_transport().open_session()
        channel.exec_command(cmd)
        return True
        # stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        #
        # err_content = stderr.read()
        # print(err_content)
        # if len(err_content) == 0:
        #     return True
        #
        # return False



    def test_tag(self, path, batch_id):
        tag_file = str()
        if self.OS == 'linux':
            tag_file = os.path.join(path, batch_id).replace('\\', '/')
        else:
            raise Exception("OS not supported")
        stdin, stdout, stderr = self.ssh_client.exec_command('test -e %s; echo $?' % tag_file)
        print('test -e %s; echo $?' % tag_file)

        ret = stdout.read()
        # ret = str(ret).strip().strip('\n').strip()
        ret = str(ret)
        print(ret)


        if '0' in ret:
            return True
        return False
