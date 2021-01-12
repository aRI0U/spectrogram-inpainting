import unittest
import model.encoders


class TestEncoders(unittest.TestCase):

	"""
		Test the encoders
	"""

	def test_mnist_encoder(self):

		"""
			Create a dummy MNIST Encoder and test its functionalities
		"""
		
		encoder = model.encoders.MNISTEncoder(in_channels=10, out_channels=10)
		for idx, (name, m) in enumerate(encoder.named_children()):
			# print(f'({idx}) -> {name}')
			self.assertTrue(m.training)