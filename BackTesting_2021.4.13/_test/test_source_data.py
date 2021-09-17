
from dc.source_data import *


def test_get_source_data():
    df = get_source_data_from_config('sc', 'YMH21')
    print(21, df)
    for i in range(10):
        print(i, type(df.iloc[1, i]))