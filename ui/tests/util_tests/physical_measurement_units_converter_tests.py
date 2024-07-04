import unittest
from util.physical_measurement_units_converter import PhysicalMeasurementUnitsConverter


class PhysicalMeasurementUnitsConverterTests(unittest.TestCase):

    def test_to_kelvin(self):
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_kelvin("0", "C"), 273.15)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_kelvin("32", "F"), 273.15)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_kelvin("300", "K"), 300)
        with self.assertRaises(ValueError):
            PhysicalMeasurementUnitsConverter.to_kelvin("100", "X")

    def test_to_seconds(self):
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_seconds("1000", "ms"), 1)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_seconds("1", "min"), 60)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_seconds("1", "s"), 1)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_seconds("1000", "ns"), 1e-6)

    def test_to_pascal(self):
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_pascal("1", "Pa"), 1)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_pascal("1", "kPa"), 1000)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_pascal("1", "psi"), 6894.76)

    def test_to_cubic_meters(self):
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_cubic_meters("1000", "cm³"), 1e-3)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_cubic_meters("1", "m³"), 1)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_cubic_meters("1", "mm³"), 1e-9)

    def test_to_joules(self):
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_joules("1", "eV"), 1.602176634e-19)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_joules("1", "keV"), 1.602176634e-16)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_joules("1", "J"), 1)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_joules("1", "kJ"), 1000)
        self.assertAlmostEqual(PhysicalMeasurementUnitsConverter.to_joules("1", "cal"), 4.184)


if __name__ == '__main__':
    unittest.main()
