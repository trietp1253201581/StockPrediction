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

Given the following stock price predictions for the next 5 days (starting from {start_date}), and a list of related news headlines, adjust the predictions based on the potential market impact of the news.

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

def get_news(llm: LLM, ticker: str, start_date: datetime, end_date: datetime, n: int = 5) -> list[dict]:
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    prompt = GET_NEWS_PROMPT.format(n=n, ticker=ticker, start_date=start_date_str, end_date=end_date_str)
    response = llm.get_response(prompt)
    json_res = llm.extract_response(response)
    
    final_res = []
    for res in json_res['results']:
        final_res.append({
            'pub_date': datetime.strptime(res['pub_date'], "%Y-%m-%d"),
            'summary': res['summary'],
            'type': res['type'],
            'direct': res['direct']
        })
    
    return final_res

def adjust_prediction(llm: LLM, original_prediction: list[float], news: list[dict], start_date: datetime) -> dict:
    start_date_str = start_date.strftime("%Y-%m-%d")
    prompt = ADJUST_PREDICTION_PROMPT.format(original_prediction=original_prediction, 
                                             news_json=news, start_date=start_date_str)
    response = llm.get_response(prompt)
    json_res = llm.extract_response(response)
    return json_res


