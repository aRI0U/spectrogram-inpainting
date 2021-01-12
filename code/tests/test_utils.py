import unittest
from time import sleep
from utils.co2_emissions import timeit

class TestCO2Emissions(unittest.TestCase):

	def test_timeit(self):

		@timeit
		def dummy():
			sleep(1)

		dummy()

