import unittest
from util.physical_measurement_units_converter import PhysicalMeasurementUnitsConverter


class PhysicalMeasurementUnitsConverterTests(unittest.TestCase):

    def test_to_kelvin(self):
        test_cases = [
            ("0", "C", 273.15),
            ("32", "F", 273.15),
            ("300", "K", 300),
            ("100", "C", 373.15),
            ("212", "F", 373.15),
            ("0", "K", 0),
            ("-273.15", "C", 0),
            ("-459.67", "F", 0),
            ("1000", "K", 1000),
            ("-40", "C", 233.15)
        ]
        for value, unit, expected in test_cases:
            with self.subTest(value=value, unit=unit, expected=expected):
                self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_kelvin(value, unit), expected)
        with self.assertRaises(ValueError):
            PhysicalMeasurementUnitsConverter.to_kelvin("100", "X")

    def test_to_seconds(self):
        test_cases = [
            ("1000", "ms", 1),
            ("1", "min", 60),
            ("1", "s", 1),
            ("1000", "ns", 1e-6),
            ("60", "min", 3600),
            ("1000", "μs", 1e-3),
            ("0", "s", 0),
            ("2", "min", 120)
        ]
        for value, unit, expected in test_cases:
            with self.subTest(value=value, unit=unit, expected=expected):
                self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_seconds(value, unit), expected)

    def test_to_pascal(self):
        test_cases = [
            ("1", "Pa", 1),
            ("1", "kPa", 1000),
            ("1", "psi", 6894.76),
            ("100", "Pa", 100),
            ("0.1", "kPa", 100),
            ("0", "Pa", 0),
            ("0", "kPa", 0),
            ("0", "psi", 0),
            ("10", "psi", 68947.6),
            ("1000", "Pa", 1000)
        ]
        for value, unit, expected in test_cases:
            with self.subTest(value=value, unit=unit, expected=expected):
                self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_pascal(value, unit), expected)

    def test_to_cubic_meters(self):
        test_cases = [
            ("1000", "cm³", 1e-3),
            ("1", "m³", 1),
            ("1", "mm³", 1e-9),
            ("1000000", "cm³", 1),
            ("1000000", "mm³", 1e-3),
            ("0", "m³", 0),
            ("1000", "m³", 1000),
            ("0.001", "m³", 1e-3),
            ("500", "cm³", 0.0005),
            ("10", "m³", 10)
        ]
        for value, unit, expected in test_cases:
            with self.subTest(value=value, unit=unit, expected=expected):
                self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_cubic_meters(value, unit), expected)

    def test_to_joules(self):
        test_cases = [
            ("1", "eV", 1.602176634e-19),
            ("1", "keV", 1.602176634e-16),
            ("1", "J", 1),
            ("1", "kJ", 1000),
            ("1", "cal", 4.184),
            ("1000", "eV", 1.602176634e-16)
        ]
        for value, unit, expected in test_cases:
            with self.subTest(value=value, unit=unit, expected=expected):
                self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_joules(value, unit), expected)


if __name__ == '__main__':
    unittest.main()
