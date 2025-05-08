from llm.llm_caller import LLM, LLMException
from datetime import datetime

GET_NEWS_PROMPT = """
You are a financial assistant.

Find the top {n} news headlines that could have impacted the stock ticker "{ticker}" between {start_date} and {end_date}. 
These should include news related to the company itself, its sector, legal or regulatory changes, macroeconomic events, or major product announcements.

For each news item, include:
- pub_date: The date of publication in YYYY-MM-DD format
- summary: A short 2-4 sentence summary of the news
- type: The category of the news (e.g., earnings, macroeconomic, legal, product, rumor, etc.)
- direct: The expected direction of impact on the stock (choose one: "positive", "negative", "unknown")

Return the result strictly as JSON in the following format:

{{
    "results": [
    {{
      "pub_date": "...",
      "summary": "...",
      "type": "...",
      "direct": "positive" | "negative" | "unknown"
    }},
    {{
      "pub_date": "...",
      "summary": "...",
      "type": "...",
      "direct": "positive" | "negative" | "unknown"
    }},
    ...
  ]
}}

Note: Do NOT include any explanation or extra commentary.
"""

ADJUST_PREDICTION_PROMPT = """
You are a financial reasoning assistant.

Given the following stock price predictions for the next 5 days (starting from {start_date}) of stock ticker {ticker}, and a list of related news headlines, adjust the predictions based on the potential market impact of the news.

Original 5-day prediction (in USD): {original_prediction}

News headlines:
{news_json}

Each news item includes:
- pub_date: The date the news was published (in YYYY-MM-DD format)
- summary: A short summary of the news
- type: Category of the news (e.g., earnings, macroeconomic, legal, product, etc.)
- direct: Expected impact direction on the stock ("positive", "negative", "unknown")

Adjust the predicted prices to reflect the possible influence of this news. The direction can affect magnitude and direction of price change.

Output only a JSON object in the following format:

{{
  "predicted": [<adjusted_price_day_1>, <adjusted_price_day_2>, <adjusted_price_day_3>, <adjusted_price_day_4>, <adjusted_price_day_5>],
  "explain": "Short explanation (2-4 sentences) of how the adjustment was made based on the news."
}}

Notes:
- Do not include any additional comments or explanation outside the JSON.
- Do not search any diffirent information and must use only the given original prediction and news.
"""

PERCENT_ADJUST_PROMPT = """
You are a financial analyst LLM. You are given a list of news articles related to stock ticker {ticker} before a prediction window.

Each news item has:
- pub_date: the date the news was published
- summary: the content or summary of the news
- type: the category of the news (e.g., earnings, legal, product, macro)
- direct: whether the news is likely positive, negative, or unknown for the stock

Here is the list of news:

{news_json} 

Based on these news items, estimate the expected market direction for this stock over the next 5 trading days, from {start_date}. 
For each day, provide your predict of:
- direction: "positive", "negative", or "neutral"
- level: "mild", "moderate", or "strong"

Return your response strictly in the following JSON format with no explanations or notes outside the format:

{{
   "predicted": [
       {{"direction": "...", "level": "..."}},
       {{"direction": "...", "level": "..."}},
       {{"direction": "...", "level": "..."}},
       {{"direction": "...", "level": "..."}},
       {{"direction": "...", "level": "..."}}
   ],
   "explain": "Short explanation of the reasoning behind your 5-day forecast."
}}
Notes:
- Do not include any additional comments or explanation outside the JSON.
- Do not search any diffirent information and must use only the given news.
"""

class LLMPredictor:
    def __init__(self, llm_caller: LLM):
        self.llm_caller = llm_caller
        self.direction_map = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
        self.level_map = {
            'mild': 1.0,
            'moderate': 3.0,
            'strong': 5.0
        }
    
    def get_news(self, ticker: str, start: datetime, end: datetime, n: int = 5):
        start_date_str = start.strftime("%Y-%m-%d")
        end_date_str = end.strftime("%Y-%m-%d")
        prompt = GET_NEWS_PROMPT.format(
            n=n, ticker=ticker, start_date=start_date_str, end_date=end_date_str
        )
        response = self.llm_caller.get_response(prompt)
        json_res = self.llm_caller.extract_response(response)

        final_res = []
        for res in json_res["results"]:
            final_res.append(
                {
                    "pub_date": datetime.strptime(res["pub_date"], "%Y-%m-%d"),
                    "summary": res["summary"],
                    "type": res["type"],
                    "direct": res["direct"],
                }
            )

        return final_res
    
    def direct_adjust(self, ticker: str, original_prediction: list[float], news: list[dict], start_date: datetime):
        start_date_str = start_date.strftime("%Y-%m-%d")
        prompt = ADJUST_PREDICTION_PROMPT.format(
            ticker=ticker,
            original_prediction=original_prediction,
            news_json=news,
            start_date=start_date_str,
        )
        response = self.llm_caller.get_response(prompt)
        json_res = self.llm_caller.extract_response(response)
        predicted = [float(x) for x in json_res['predicted']]
        explain = json_res['explain']
        return predicted, explain
    
    def percent_adjust(self, ticker: str, last_day: float,
                       original_prediction: list[float], news: list[dict], 
                       start_date: datetime, confidence_rate: float=0.5):
        start_date_str = start_date.strftime("%Y-%m-%d")
        prompt = PERCENT_ADJUST_PROMPT.format(
            ticker=ticker,
            news_json=news,
            start_date=start_date_str,
        )
        response = self.llm_caller.get_response(prompt)
        json_res = self.llm_caller.extract_response(response)
        predicted = []
        for dir_lev in json_res['predicted']:
            direction = dir_lev['direction']
            level = dir_lev['level']
            
            rate = self.direction_map.get(direction, 0.0) * self.level_map.get(level, 0.0)
            predicted.append(last_day * (1.0 + rate))
            
        for i in range(len(predicted)):
            predicted[i] = predicted[i] * confidence_rate + original_prediction[i] * (1 - confidence_rate)
                    
        return predicted, json_res['explain']