#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 对数据库进行操作
# Author: Tayii
# Data : 2021/02/22
# ----------------------------------------------------
import json
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Union
import pandas as pd
import pymysql
from queue import Queue

from datahub.fmt import DataBase


@dataclass
class DbConfig:
    driver_name: str
    username: str
    password: str
    host: str
    port: int
    database: int
    query: Any = None


class DB(DataBase):
    """
    数据库操作
    """

    def __init__(self, name: str,
                 config_file: str,  # 配置文件 json格式
                 pool_size: int = 10,  # 数据池最大连接数
                 ):
        DataBase.__init__(self, name)

        with open(config_file, 'r') as f:
            c_ = json.load(f)
        self.config = DbConfig(driver_name=c_['driver'],
                               username=c_['username'],
                               password=c_['password'],
                               host=c_['host'],
                               port=c_['port'],
                               database=c_['database'],
                               query={'charset': 'utf8'},
                               )

        self.__pool = Queue(pool_size)  # queue用做pool
        for _ in range(pool_size):
            self.__pool.put(self.__create_conn())

    def __create_conn(self) -> pymysql.connect:
        """创建 connection"""
        return pymysql.connect(
            user=self.config.username,
            passwd=self.config.password,
            host=self.config.host,
            port=self.config.port,
            db=self.config.database,
            charset=self.config.query['charset'],
        )

    def __get_conn(self) -> pymysql.connect:
        return self.__pool.get()

    def pool_conn(func):
        """
        @装饰器
        用数据池的connection来执行各种操作(当传入conn==None时)
        自动分配 自动回收
        """
        @wraps(func)  # 把原函数的元信息拷贝到装饰器里面的 func函数中
        def wraaper(self, *args, **kwargs):
            conn: pymysql.connect = kwargs.get('conn', self.__get_conn())
            kwargs['conn'] = conn
            try:
                with conn:
                    return func(self, *args, **kwargs)
            except Exception as e:
                raise e

            finally:
                self.__pool.put(conn)  # 回收

        return wraaper

    def insert_df(self,
                  df: Union[pd.DataFrame, pd.Series],
                  table_name: str,
                  ) -> None:
        """Insert pandas.DataFrame(df) into table ``table_name``"""

    def insert(self):
        """ 插入 """
        cur = self.conn.cursor()

    @pool_conn
    def fetch_all(self,
                  sql: str,  # sql执行语句
                  conn: pymysql.connect = None,  # 数据库连接
                  ):
        """获取 查询到的全部记录"""
        with conn.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()

    @pool_conn
    def fetch_one(self,
                  sql: str,  # sql执行语句
                  conn: pymysql.connect = None,  # 数据库连接
                  ):
        """获取单条记录"""
        with conn.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchone()

    @pool_conn
    def insert(self,
               sql: str,  # sql执行语句
               conn: pymysql.connect = None,  # 数据库连接
               ):
        with conn.cursor() as cursor:
            cursor.execute(sql)
        # 如果没有设置autocommit, must commit to save changes
        conn.commit()

    @property
    def pool_size(self):
        return self.__pool.qsize()


if __name__ == '__main__':
    db = DB('db', 'db_config.json')

    print(db, db.fetch_all)
    ret = db.fetch_all(sql='SELECT * from object')
    print(ret)  # desc, rows, rowcount
    ret = db.fetch_one(sql='SELECT * from object')
    print(ret)  # desc, rows, rowcount
    # print(rowcount, rows)
