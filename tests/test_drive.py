import unittest,math
from warehouse_robot.drive import wheel_speeds,wrap,corners,WHEEL_RADIUS,TRACK_WIDTH

class DriveTests(unittest.TestCase):
    def test_wheel_commands(self):
        a,b=wheel_speeds(.24,0)
        self.assertAlmostEqual(a,2);self.assertAlmostEqual(b,2)
        a,b=wheel_speeds(0,1)
        self.assertLess(a,0);self.assertGreater(b,0)
        self.assertAlmostEqual(WHEEL_RADIUS*(b-a)/TRACK_WIDTH,1)
    def test_safe_corner_preservation(self):
        route=[(0,0),(0,.1),(0,.2),(.1,.2),(.2,.2)]
        self.assertEqual(corners(route),[(0,0),(0,.2),(.2,.2)])
        self.assertAlmostEqual(wrap(3*math.pi),math.pi)
