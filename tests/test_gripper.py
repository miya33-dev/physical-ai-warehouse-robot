"""Prevent accidentally reintroducing an attachment constraint into contact grasping."""
from pathlib import Path
import unittest
import mujoco as mj

class GripperModelTests(unittest.TestCase):
    def test_no_attachment_constraint_and_finite_force(self):
        root=Path(__file__).resolve().parents[1]
        m=mj.MjModel.from_xml_path(str(root/'assets/warehouse.xml'))
        self.assertEqual(m.neq,0)
        for name in ['left_grip','right_grip']:
            a=mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,name)
            self.assertGreaterEqual(a,0)
            self.assertTrue(m.actuator_forcelimited[a])
            self.assertLessEqual(m.actuator_forcerange[a,1],15)
