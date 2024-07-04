import unittest
from PyQt5.QtGui import QValidator
from PyQt5.QtCore import QObject
from field_validators import CustomIntValidator


class CustomIntValidatorTests(unittest.TestCase):

    def setUp(self):
        self.parent = QObject()
        self.validator = CustomIntValidator(0, 100, self.parent)

    def test_valid_input(self):
        state, _, _ = self.validator.validate("50", 0)
        self.assertEqual(state, QValidator.Acceptable)

    def test_below_range_input(self):
        state, _, _ = self.validator.validate("-1", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_above_range_input(self):
        state, _, _ = self.validator.validate("101", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_empty_input(self):
        state, _, _ = self.validator.validate("", 0)
        self.assertEqual(state, QValidator.Intermediate)

    def test_non_integer_input(self):
        state, _, _ = self.validator.validate("abc", 0)
        self.assertEqual(state, QValidator.Invalid)

    def test_intermediate_input(self):
        state, _, _ = self.validator.validate("5a", 0)
        self.assertEqual(state, QValidator.Invalid)


if __name__ == '__main__':
    unittest.main()
