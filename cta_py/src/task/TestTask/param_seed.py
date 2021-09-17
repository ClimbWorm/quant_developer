import numpy as np

param_list = [
    {
        "name" : "zzg_num",
        "order" : 1,
        "type" : "float",
        "values" : [0.382,
                    0.5,
                    0.618,
                    0.782,
                    0.886,
                    1.0,
                    1.236,
                    1.5,
                    2.0
        ]
    },

    {
        "name" : "ema_num",
        "order" : 2,
        "type" : "int",
        "values" : [i for i in range(13,57,2)]
    },

    {
        "name" : "multiplier",
        "order" : 2,
        "type" : "float",
        "values" : np.arange(0.1,2.7,0.2)
    },
]

# Todo 可以修改参数限制条件
