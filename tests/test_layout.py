import unittest
from warehouse_robot.layout import random_layout,validate_layout

class LayoutTests(unittest.TestCase):
    def test_seeds_reproduce_and_cover_both_color_orders(self):
        orders=set()
        for seed in range(100):
            a=random_layout(seed)
            self.assertEqual(a,random_layout(seed))
            self.assertEqual(a,validate_layout(a))
            orders.add(a['red'][1]>a['blue'][1])
        self.assertEqual(orders,{False,True})
        self.assertNotEqual(random_layout(0),random_layout(1))
    def test_reject_invalid_layouts(self):
        valid={'red':[-2.0,-.4],'blue':[-1.9,.4]}
        self.assertEqual(validate_layout(valid),valid)
        for bad in [None,{'red':[-2,-.4]}, {**valid,'green':[-2,0]},
                    {**valid,'red':[-3,0]}, {**valid,'red':[-2,float('nan')]},
                    {**valid,'red':[-2,.4]}, {**valid,'red':[True,-.4]}]:
            with self.assertRaises(ValueError):validate_layout(bad)
        for bad in [-1,True,1.5]:
            with self.assertRaises(ValueError):random_layout(bad)
