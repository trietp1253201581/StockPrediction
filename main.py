from llm.llm_support import LLMReflector
from llm.llm_caller import GoogleAIStudioLLM
from datetime import datetime

llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='llm/config.json',
                              runtime_config='llm/llm_runtime_config.json')

predicted = [
    [500, 501, 498, 486, 490],
    [300, 287, 299, 301, 302],
    [315, 316, 312, 330, 320]
]

actual = [
    [501, 498, 494, 480, 475],
    [300, 293, 290, 290, 285],
    [320, 315, 300, 305, 302]
]

reflector = LLMReflector(llm_model)

reflector.fit('AAPL', predicted, actual, None)

for ref in reflector.reflections['AAPL']:
    print(ref)
    
print(reflector.summary['AAPL'])

print(reflector.adjust('AAPL', [400, 405, 403, 402, 402], None, 'all'))

