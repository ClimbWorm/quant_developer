from functools import wraps
import time

def timeit(func):
	"""
	@装饰器 计算func消耗的时间
	PS: 多装饰器时 放在外层
	Args:
		func: 被装饰的函数
	"""
	@wraps(func) # 把原始函数的__name__等属性复制到warpper()函数中
	def wrapper(*args, **kwargs):
		start = time.time()
		ret = func(*args, **kwargs)
		cost = time.time() - start
		print('.' * 30 + '[{}]  cost {:.2f}\'ms'.format(func.__name__, cost*1000))
		return ret
	return wrapper