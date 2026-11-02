import unittest
from warehouse_robot.language import IntentModel
from warehouse_robot.planning import plan,blocked

class CoreTests(unittest.TestCase):
    def test_language_valid_and_rejection(self):
        m=IntentModel()
        for text,color,dest in [
            ('赤い荷物を出荷場所へ届けて','red','shipping'),
            ('保管場所に青い箱を運んで','blue','storage'),
            ('bring red box to storage','red','storage'),
            ('take blue box to shipping','blue','shipping')]:
            x=m.predict(text); self.assertEqual((x['color'],x['destination']),(color,dest))
        for text in ['運んで','赤い箱を玄関へ運んで','赤と青を出荷へ運んで','緑の箱を保管へ運んで','赤い箱を出荷へ運ばないで','赤い箱を出荷へ','青い箱を保管と出荷へ運んで']:
            with self.assertRaises(ValueError,msg=text): m.predict(text)
    def test_path_routes_around_inflated_obstacle(self):
        for a,b in [((-2.6,0),(1.9,-1.25)),((-2.6,.4),(1.9,1.25)),((1.9,-1.2),(-2.6,.4))]:
            path=plan(a,b)
            self.assertTrue(all(not blocked(x,y) for x,y in path))
            for p,q in zip(path,path[1:]):
                self.assertLessEqual(abs(p[0]-q[0])+abs(p[1]-q[1]),.100001)
        with self.assertRaises(ValueError): plan((0,0),(1.9,1.25))

if __name__=='__main__': unittest.main()
