import sys
import paramiko
import multiprocessing

def getConnection(ip,username,password,command,port):
    """
    :param ip: 服务器的ip
    :param username:  服务器的用户名称
    :param password:  服务器的密码
    :param CMD:  服务器的命令
    :param port:  服务器的端口
    """
    ssh = paramiko.SSHClient()
    policy = paramiko.AutoAddPolicy()
