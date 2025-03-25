import unittest
from PyQt5.QtGui import QValidator
from PyQt5.QtCore import QObject
from field_validators import CustomDoubleValidator


class CustomDoubleValidatorTests(unittest.TestCase):
    def setUp(self):
        self.parent = QObject()
        self.validator = CustomDoubleValidator(0.0, 100.0, 2, self.parent)

    def test_valid_input(self):
        state, _, _ = self.validator.validate("50.5", 0)
        self.assertEqual(state, QValidator.Acceptable)

    def test_below_range_input(self):
        state, _, _ = self.validator.validate("-1.0", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_above_range_input(self):
        state, _, _ = self.validator.validate("101.0", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_empty_input(self):
        state, _, _ = self.validator.validate("", 0)
        self.assertEqual(state, QValidator.Intermediate)

    def test_non_double_input(self):
        state, _, _ = self.validator.validate("abc", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_valid_zero_input(self):
        state, _, _ = self.validator.validate("0", 0)
        self.assertEqual(state, QValidator.Acceptable)

    def test_valid_decimal_places(self):
        state, _, _ = self.validator.validate("50.55", 0)
        self.assertEqual(state, QValidator.Acceptable)

    def test_invalid_decimal_places(self):
        state, _, _ = self.validator.validate("50.555", 0)
        self.assertEqual(state, QValidator.Invalid)


if __name__ == "__main__":
    unittest.main()
