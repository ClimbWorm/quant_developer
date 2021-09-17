from datahub.prepare_source_data import *
from multiprocessing import Queue

bg = BarGenerator2( Queue(),
source_config=YMH21_1_SC,
)


def test_bar_generate():
    assert False
