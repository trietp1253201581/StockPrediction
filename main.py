from llm.llm_support import get_news, adjust_prediction
from llm.llm_caller import GoogleAIStudioLLM
from datetime import datetime

llm_model = GoogleAIStudioLLM(model='gemini-2.0-flash', timeout=(60,600), 
                              core_config='llm/config.json',
                              runtime_config='llm/llm_runtime_config.json')

news = get_news(llm_model, 'AAPL', datetime(2025, 1, 13), datetime(2025, 1, 17), 4)

print(news)

print(adjust_prediction(llm_model, [394, 402, 403, 410, 388], news, datetime(2025, 1, 13)))

