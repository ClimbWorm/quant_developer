


if __name__ == '__main__':


    import paramiko

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname='192.168.100.226',port=22,username='protondb',password='123')
    #执行命令
    stdin, stdout, stderr = ssh_client.exec_command('touch helloworld')



    # 连接远程服务器执行上传下载
    ftp_client = ssh_client.open_sftp()

    # 将本地文件上传至服务器
    ftp_client.put(r'F:\pythonHistoricalTesting\pythonHistoricalTesting\learn_flask.py','/home/protondb/Iron_Musk/work/test.py')


    # 将文件从服务器拿到本地
    ftp_client.get('/home/protondb/Iron_Musk/work/test.py',r'F:\pythonHistoricalTesting\pythonHistoricalTesting\test.py')
    ftp_client.close()

    ssh_client.close()