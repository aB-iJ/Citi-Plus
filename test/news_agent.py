import requests
from bs4 import BeautifulSoup
import json
import re
import datetime

class DeepSeekAnalyzer:
    """
    负责调用 DeepSeek API 进行金融新闻情感分析
    """
    def __init__(self, api_key):
        self.api_key = api_key
        # DeepSeek API 端点 (请以官方文档为准，通常是兼容 OpenAI 格式)
        self.base_url = "https://api.deepseek.com/chat/completions" 
        
    def analyze_sentiment(self, headlines):
        """
        输入: 新闻标题列表
        输出: -1 (极负面) 到 1 (极正面) 的浮点数
        """
        if not headlines:
            return 0.0

        # 构造 Prompt
        news_text = "\n".join([f"- {h}" for h in headlines])
        prompt = f"""
        你是一个专业的原油期货市场分析师。请分析以下今日新闻标题对于 WTI 原油价格的短期（明日）影响：
        
        {news_text}
        
        请综合考虑供需关系、地缘政治、美元汇率等因素。
        请仅输出一个介于 -1.0 到 1.0 之间的数值。
        -1.0 代表极度利空（如经济衰退、产量大增）。
        0.0 代表无影响或多空抵消。
        1.0 代表极度利多（如战争爆发、OPEC减产）。
        
        格式要求：不要输出任何解释文字，只输出这一个数字。
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "deepseek-chat", # 或 deepseek-reasoner
            "messages": [
                {"role": "system", "content": "You are a specialized financial sentiment analyzer for Crude Oil markets."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1 #以此降低随机性
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result_json = response.json()
            content = result_json['choices'][0]['message']['content'].strip()
            
            # 提取数字 (防止模型偶尔废话)
            match = re.search(r"-?\d+(\.\d+)?", content)
            if match:
                score = float(match.group())
                # 再次截断以防万一
                return max(-1.0, min(1.0, score))
            else:
                print(f"DeepSeek response format warning: {content}")
                return 0.0
                
        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return 0.0

class NewsCrawler:
    """
    简易新闻爬虫。
    注意：彭博/路透等网站有极强的反爬虫防护。
    建议爬取 'Investing.com' 或 'OilPrice.com' 等聚合类信息，或者利用 RSS Feed。
    """
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def fetch_specific_date_news(self, date_str):
        """
        [新功能] 针对特定日期抓取历史新闻。
        使用 DuckDuckGo HTML 搜索（模拟浏览器），搜索 "crude oil news yyyy-mm-dd"。
        旨在解决列表页无法回溯很久的问题。
        """
        headlines = []
        try:
            # 构造搜索查询：site限定在权威网站，加上日期
            query = f"crude oil news wti {date_str} site:investing.com"
            # DuckDuckGo HTML 搜索入口
            url = f"https://html.duckduckgo.com/html/?q={query}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取搜索结果标题
                results = soup.find_all('a', class_='result__a')
                
                for res in results[:3]: # 每天只取最相关的 3 条，避免噪音
                    title = res.get_text().strip()
                    # 简单过滤广告
                    if len(title) > 20 and "..." not in title[-3:]:
                        headlines.append(title)
            
            print(f"  [Crawler] Found {len(headlines)} news for {date_str}")
            # 简单的防封禁延时
            import time
            time.sleep(1)
            return headlines
            
        except Exception as e:
            print(f"  [Crawler] Error searching for {date_str}: {e}")
            return []

    def crawl_last_n_days(self, n=90, output_file="crawled_news.json"):
        """
        爬取过去 n 天的新闻并保存。
        如果本地已有部分数据，会跳过已存在的日期（增量更新）。
        """
        import os
        from config import config # 动态导入检查配置
        
        # 1. 检查配置：如果用户选择只用现有新闻，则不联网抓取
        if hasattr(config, 'USE_EXISTING_NEWS') and config.USE_EXISTING_NEWS:
             print("[Config] USE_EXISTING_NEWS=True. Skipping new web crawling.")
             if os.path.exists(output_file):
                try:
                    with open(output_file, "r", encoding='utf-8') as f:
                        return json.load(f)
                except:
                    return {}
             else:
                 return {}

        # 2. 读取现有数据
        news_data = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding='utf-8') as f:
                    news_data = json.load(f)
            except:
                print("Read JSON failed, starting fresh.")
                news_data = {}
        
        # 3. 生成日期列表 (从今天倒推 n 天)
        target_dates = []
        for i in range(n):
            dt = datetime.datetime.now() - datetime.timedelta(days=i)
            target_dates.append(dt.strftime('%Y-%m-%d'))
            
        # 4. 遍历抓取
        updated = False
        print(f"Starting historical crawl for {n} days...")
        
        for date_str in target_dates:
            if date_str in news_data and len(news_data[date_str]) > 0:
                # 已有数据，跳过
                continue
            
            print(f"Fetching news for {date_str}...")
            headlines = self.fetch_specific_date_news(date_str)
            
            # [新功能] 如果抓取到空数据 (由于反爬虫等原因)，尝试从备用源或 yfinance 获取最近数据
            if not headlines:
                 headlines = self.fetch_backup_news(date_str)

            if headlines:
                news_data[date_str] = headlines
                updated = True
        
        # 5. 保存
        if updated:
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(news_data, f, ensure_ascii=False, indent=4)
            print(f"Updated news data saved to {output_file}")
            
        return news_data
        
    def fetch_backup_news(self, date_str):
        """
        备用获取策略: 如果 DuckDuckGo 失败，尝试 yfinance 获取最近新闻
        (仅对最近几天有效，历史数据仍可能为空)
        """
        try:
            import yfinance as yf
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            # 只有当请求日期是最近2天时，yfinance 才可能返回相关新闻列表
            # 这是一个非常简单的 heuristic
            if date_str >= (datetime.datetime.now() - datetime.timedelta(days=2)).strftime('%Y-%m-%d'):
                print("  [Backup] Trying yfinance news...")
                ticker = yf.Ticker("CL=F")
                news_list = ticker.news
                headlines = []
                for item in news_list:
                    # yfinance news item 包含 'title' 和 'providerPublishTime'
                    # 这里简化处理，全部返回，主要为了保证有数据
                    headlines.append(item.get('title', ''))
                return headlines[:5]
        except:
            pass
        return []

    # 兼容旧接口，直接返回最新一天
    def fetch_investing_com_news(self):
        # 默认只抓取最近 1 天 (今天)
        return self.crawl_last_n_days(n=1)


if __name__ == "__main__":
    # 测试代码
    # 1. 爬取
    crawler = NewsCrawler()
    print("正在测试爬虫...")
    news = crawler.fetch_investing_com_news()
    
    if news:
        print("--- 最新新闻 ---")
        for i, n in enumerate(news):
            print(f"{i+1}. {n}")
            
        # 2. 分析 (需要您填入真实的 Key)
        print("\n若要测试 AI 分析功能，请取消注释代码中的 DeepSeekAnalyzer 相关行并填入您的 API Key。")
        # YOUR_API_KEY = "sk-xxxxxxxx"
        # analyzer = DeepSeekAnalyzer(api_key=YOUR_API_KEY)
        # score = analyzer.analyze_sentiment(news)
        # print(f"\nAI 情感评分: {score}")
    else:
        print("未找到新闻。请检查选择器或网络连接。")
