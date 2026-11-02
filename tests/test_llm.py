import unittest
from unittest.mock import patch
import json
from warehouse_robot.llm_language import LocalLLM

class FakeResponse:
    def __init__(self,obj):self.payload=json.dumps(obj).encode()
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,*args):return self.payload

class LLMTests(unittest.TestCase):
    def test_strict_schema_validation(self):
        valid={'decision':'deliver','target_color':'red','destination':'shipping','message':'配送'}
        self.assertEqual(LocalLLM.validate(valid),valid)
        for extra in [{'target_color':'green'},{'destination':'unknown'},{'decision':'execute_code'},{'message':None},{'code':'print(1)'}]:
            with self.assertRaises(ValueError):LocalLLM.validate({**valid,**extra})
    def test_model_decision_and_independent_guard(self):
        valid={'decision':'deliver','target_color':'red','destination':'shipping','message':'配送'}
        def response(obj):return FakeResponse({'done':True,'done_reason':'stop','message':{'content':json.dumps(obj)}})
        with patch('urllib.request.urlopen',return_value=response(valid)):
            self.assertEqual(LocalLLM().predict('赤い箱を出荷場所へ運んで')['color'],'red')
        for text in ['赤い箱を出荷場所へ運ばないで','もし赤い箱なら出荷へ運んで','赤い箱を運んで','赤い箱を出荷へ運べますか？']:
            with patch('urllib.request.urlopen',return_value=response(valid)):
                with self.assertRaises(ValueError):LocalLLM().predict(text)
        with patch('urllib.request.urlopen',return_value=response({**valid,'decision':'clarify'})):
            with self.assertRaises(ValueError):LocalLLM().predict('箱を運んで')
    def test_no_silent_fallback(self):
        import urllib.error
        with patch('urllib.request.urlopen',side_effect=urllib.error.URLError('offline')):
            with self.assertRaisesRegex(ValueError,'No fallback'):LocalLLM().predict('赤い箱を出荷へ運んで')
