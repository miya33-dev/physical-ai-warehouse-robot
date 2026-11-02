# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from warehouse_robot.planning import plan,RES
from warehouse_robot.presentation import pixel
class PlanningTraceTests(unittest.TestCase):
    def test_trace_is_observational_and_reaches_goal(self):
        start=(-2.6,.3);goal=(1.9,-1.25);trace={}
        self.assertEqual(plan(start,goal),plan(start,goal,trace=trace))
        self.assertEqual(trace['path'],[list(p) for p in plan(start,goal)])
        self.assertEqual(trace['events'][0]['current'],[round(v/RES) for v in start])
        self.assertEqual(trace['events'][-1]['current'],[round(v/RES) for v in goal])
        self.assertEqual(trace['events'][-1]['g'],len(trace['path'])-1)
    def test_floor_projection_matches_camera_axes(self):
        self.assertEqual(pixel(0,0),(320.,240.))
        self.assertGreater(pixel(1,0)[0],320)
        self.assertLess(pixel(0,1)[1],240)
