#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# ----------------------------------------------------
# 
# Author: Tayii
# Data : 2020/12/6 20:17
# ----------------------------------------------------
import datetime as dt
import json
import sqlalchemy as sa
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker
from typing import List, Mapping, Optional, Sequence, Union

import pandas as pd


class DBInterface(object):
    """Database Interface Base Class"""

    def __init__(self):
        pass

    def create_table(self, table_name: str, table_info: Mapping[str, str]) -> None:
        """Create table named ``table_name`` with column name and type specified in ``table_info``"""
        raise NotImplementedError()

    def drop_all_tables(self) -> None:
        """[CAUTION] Drop *ALL TABLES AND THEIR DATA* in the database"""
        raise NotImplementedError()

    def purge_table(self, table_name: str) -> None:
        """[CAUTION] Drop *ALL DATA* in the table"""
        raise NotImplementedError()

    def insert_df(self, df: Union[pd.DataFrame, pd.Series], table_name: str) -> None:
        """Insert pandas.DataFrame(df) into table ``table_name``"""
        raise NotImplementedError()

    def update_df(self, df: Union[pd.DataFrame, pd.Series], table_name: str) -> None:
        """Update pandas.DataFrame(df) into table ``table_name``"""
        raise NotImplementedError()

    def update_compact_df(self, df: pd.Series, table_name: str, old_df: pd.Series = None) -> None:
        """Update new information from df to table ``table_name``"""
        raise NotImplementedError()

    def get_latest_timestamp(self, table_name: str, column_condition: (str, str) = None) -> Optional[dt.datetime]:
        """Get the latest timestamp from records in ``table_name``"""
        raise NotImplementedError()

    def read_table(self, table_name: str, columns: Sequence[str] = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Read data from ``table_name``"""
        raise NotImplementedError()

    def get_all_id(self, table_name: str) -> Optional[List[str]]:
        """Get all stocks in a table"""
        raise NotImplementedError()

    def get_column(self, table_name: str, column_name: str) -> Optional[List]:
        """Get a column from a table"""
        raise NotImplementedError()

    def exist_table(self, table_name: str) -> bool:
        """Check if ``table_name`` exists in the database"""
        raise NotImplementedError()

    def get_columns_names(self, table_name: str) -> List[str]:
        """Get column names of a table"""
        raise NotImplementedError()

    def get_table_primary_keys(self, table_name: str) -> Optional[List[str]]:
        """Get primary keys of a table"""
        raise NotImplementedError()

    def get_table_names(self) -> List[str]:
        """List ALL tables in the database"""
        raise NotImplementedError()

    def get_column_min(self, table_name: str, column: str):
        raise NotImplementedError()

    def get_column_max(self, table_name: str, column: str):
        raise NotImplementedError()


def prepare_engine(config_file: str) -> sa.engine.Engine:
    """Create sqlalchemy engine from config file"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    url = URL(drivername=config['driver'],
              host=config['host'],
              port=config['port'],
              database=config['database'],
              username=config['username'],
              password=config['password'],
              query={'charset': 'utf8'},
              )
    print(url)
    return sa.create_engine(url)


if __name__ == '__main__':
    config_file = 'db_config.json'
    print(prepare_engine(config_file))

    # jicewarwick  AShareData
