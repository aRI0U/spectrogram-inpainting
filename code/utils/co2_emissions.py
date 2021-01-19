from time import time

def timeit(function):
	""" Decorator that records the running time of functions """

	def timed(*args, **kwargs):

		ts = time()
		result = function(*args, **kwargs)
		te = time()

		t = te - ts
		print(f'"{function.__name__}" took {t / 360:02.0f}:{t / 60:02.0f}:{t:02.0f} to run')

		return result

	return timed