# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from warehouse_robot.detector import select_target,pixel_to_world
class DetectorSelectionTests(unittest.TestCase):
    def test_projection_and_selection(self):
        self.assertEqual(pixel_to_world(320,240,640,480),[0.,0.])
        box={'class':'red_cube','score':.9,'bbox_xyxy':[133,191,153,211]}
        r=select_target([box],'red',640,480)
        self.assertTrue(-2.25<r['xy'][0]<-1.65)
        self.assertEqual(r['pixel_center'],[143.,201.])
    def test_missing_and_ambiguous_targets_stop(self):
        b={'class':'red_cube','score':.9,'bbox_xyxy':[133,191,153,211]}
        for detections in [[],[{**b,'score':.1}],[b,b],[{**b,'class':'blue_cube'}],[{**b,'bbox_xyxy':[0,0,20,20]}]]:
            with self.assertRaises(ValueError):select_target(detections,'red',640,480)
