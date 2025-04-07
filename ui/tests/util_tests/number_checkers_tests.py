import unittest
import random
import string
from util.number_checkers import *


class NumberCheckersTests(unittest.TestCase):
    def generate_random_strings(self, length=10):
        """Generate a random string of fixed length."""
        letters = string.ascii_letters + string.digits + string.punctuation
        return "".join(random.choice(letters) for i in range(length))

    def test_is_real_number(self):
        for _ in range(100000):
            value = random.choice(
                [
                    str(random.uniform(-1e308, 1e308)),  # Valid real number
                    self.generate_random_strings(),  # Random string
                    str(random.randint(-1e6, 1e6)),  # Valid integer
                    random.choice(["NaN", "Infinity", "-Infinity"]),  # Special floats
                ]
            )
            try:
                expected = float(value) is not None
            except ValueError:
                expected = False
            result = is_real_number(value)
            if result != expected:
                print(
                    f"test_is_real_number(): Failed value: {value}, Expected: {expected}, Got: {result}"
                )
            self.assertEqual(is_real_number(value), expected)

    def test_is_positive_real_number(self):
        for _ in range(100000):
            value = random.choice(
                [
                    str(random.uniform(0, 1e308)),  # Valid positive real number
                    str(random.uniform(-1e308, 0)),  # Negative real number
                    self.generate_random_strings(),  # Random string
                    str(random.randint(-1e6, 1e6)),  # Valid integer
                    random.choice(["NaN", "Infinity", "-Infinity"]),  # Special floats
                ]
            )
            try:
                expected = float(value) >= 0 and not value in [
                    "NaN",
                    "Infinity",
                    "-Infinity",
                ]
            except ValueError:
                expected = False
            result = is_positive_real_number(value)
            if result != expected:
                print(
                    f"test_is_positive_real_number(): Failed value: {value}, Expected: {expected}, Got: {result}"
                )
            self.assertEqual(is_positive_real_number(value), expected)

    def test_is_positive_natural_number(self):
        for _ in range(100000):
            value = random.choice(
                [
                    str(random.randint(1, 1e6)),  # Valid positive natural number
                    str(random.randint(-1e6, 0)),  # Negative integer
                    self.generate_random_strings(),  # Random string
                    str(random.uniform(0, 1e6)),  # Real number
                    str(random.uniform(-1e6, 0)),  # Negative real number
                ]
            )
            try:
                expected = int(value) > 0
            except ValueError:
                expected = False
            result = is_positive_natural_number(value)
            if result != expected:
                print(
                    f"test_is_positive_natural_number(): Failed value: {value}, Expected: {expected}, Got: {result}"
                )
            self.assertEqual(is_positive_natural_number(value), expected)


if __name__ == "__main__":
    unittest.main()
