import unittest
from PyQt5.QtGui import QValidator, QIntValidator
from PyQt5.QtCore import QObject


class CustomIntValidator(QIntValidator):
    def __init__(self, bottom, top, parent=None):
        super().__init__(bottom, top, parent)

    def validate(self, input_str, pos):
        if not input_str:
            return (self.Intermediate, input_str, pos)

        try:
            value = int(input_str)
            if self.bottom() <= value <= self.top():
                return (self.Acceptable, input_str, pos)
            else:
                return (self.Invalid, input_str, pos)
        except ValueError:
            return (self.Invalid, input_str, pos)

        return (self.Intermediate, input_str, pos)


class TestCustomIntValidator(unittest.TestCase):

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
