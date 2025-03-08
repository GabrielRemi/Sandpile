import unittest
from .avalanche import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(add(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
