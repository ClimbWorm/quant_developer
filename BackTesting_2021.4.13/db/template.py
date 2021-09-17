#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 数据库 接口模板
#
# Author: Tayii
# Data : 2020/12/02
# ----------------------------------------------------
from abc import abstractmethod
import datetime
from typing import Union, Sequence, Optional, List, Mapping
import pandas as pd

from datahub.fmt import DataBase


class DBInterface(DataBase):
    """Database Interface Base Class"""

    def __init__(self, name: str):
        DataBase.__init__(self, name=name)

    # @abstractmethod
    # def create_table(self, table_name: str, table_info: Mapping[str, str]) -> None:
    #     """Create table named ``table_name`` with column name and type specified in ``table_info``"""
    #
    # @abstractmethod
    # def drop_all_tables(self) -> None:
    #     """[CAUTION] Drop *ALL TABLES AND THEIR DATA* in the database"""
    #
    # @abstractmethod
    # def purge_table(self, table_name: str) -> None:
    #     """[CAUTION] Drop *ALL DATA* in the table"""
    #
    # @abstractmethod
    # def insert_df(self, df: Union[pd.DataFrame, pd.Series], table_name: str) -> None:
    #     """Insert pandas.DataFrame(df) into table ``table_name``"""
    #
    # @abstractmethod
    # def update_df(self, df: Union[pd.DataFrame, pd.Series], table_name: str) -> None:
    #     """Update pandas.DataFrame(df) into table ``table_name``"""
    #
    # @abstractmethod
    # def update_compact_df(self, df: pd.Series, table_name: str, old_df: pd.Series = None) -> None:
    #     """Update new information from df to table ``table_name``"""
    #
    # @abstractmethod
    # def get_latest_timestamp(self, table_name: str, column_condition: (str, str) = None) -> Optional[datetime.datetime]:
    #     """Get the latest timestamp from records in ``table_name``"""
    #
    # @abstractmethod
    # def read_table(self, table_name: str, columns: Sequence[str] = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    #     """Read data from ``table_name``"""
    #
    # @abstractmethod
    # def get_all_id(self, table_name: str) -> Optional[List[str]]:
    #     """Get all stocks in a table"""
    #
    # @abstractmethod
    # def get_column(self, table_name: str, column_name: str) -> Optional[List]:
    #     """Get a column from a table"""
    #
    # @abstractmethod
    # def exist_table(self, table_name: str) -> bool:
    #     """Check if ``table_name`` exists in the database"""
    #
    # @abstractmethod
    # def get_columns_names(self, table_name: str) -> List[str]:
    #     """Get column names of a table"""
    #
    # @abstractmethod
    # def get_table_primary_keys(self, table_name: str) -> Optional[List[str]]:
    #     """Get primary keys of a table"""
    #
    # @abstractmethod
    # def get_table_names(self) -> List[str]:
    #     """List ALL tables in the database"""
    #
    # @abstractmethod
    # def get_column_min(self, table_name: str, column: str):
    #     """"""
    #
    # @abstractmethod
    # def get_column_max(self, table_name: str, column: str):
    #     """"""
