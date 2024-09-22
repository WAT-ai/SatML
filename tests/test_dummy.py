import unittest

class TestHelloWorld(unittest.TestCase):
    
    def test_hello_world(self):
        """A simple test to check if the string is 'Hello, World!'"""
        message = "Hello, World!"
        self.assertEqual(message, "Hello, World!")