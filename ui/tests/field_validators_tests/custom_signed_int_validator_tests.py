import unittest
from PyQt5.QtGui import QValidator
from PyQt5.QtCore import QObject
from field_validators import CustomSignedIntValidator


class CustomSignedIntValidatorTests(unittest.TestCase):

    def setUp(self):
        self.parent = QObject()
        self.validator = CustomSignedIntValidator(-100, 100, self.parent)

    def test_valid_input(self):
        state, _, _ = self.validator.validate("50", 0)
        self.assertEqual(state, QValidator.Acceptable)

    def test_negative_input(self):
        state, _, _ = self.validator.validate("-50", 0)
        self.assertEqual(state, QValidator.Acceptable)

    def test_below_range_input(self):
        state, _, _ = self.validator.validate("-101", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_above_range_input(self):
        state, _, _ = self.validator.validate("101", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_empty_input(self):
        state, _, _ = self.validator.validate("", 0)
        self.assertEqual(state, QValidator.Intermediate)

    def test_only_minus_input(self):
        state, _, _ = self.validator.validate("-", 0)
        self.assertEqual(state, QValidator.Intermediate)

    def test_non_integer_input(self):
        state, _, _ = self.validator.validate("abc", 0)
        self.assertEqual(state, QValidator.Invalid)


if __name__ == '__main__':
    unittest.main()
