import googleapiclient
from googleapiclient import discovery
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from collections.abc import Iterator
from threading import Lock
import numpy as np

PERSPECTIVE_API_KEY = ""        # Place your KEY here
QUOTA_IN_QPS = 1                # QPS>100 is optimal, see Readme.md for ways to increase API quota

class RateLimiter(Iterator):
    """Iterator that yields a value at most once every 'interval' seconds."""
    def __init__(self, interval):
        self.lock = Lock()
        self.interval = interval
        self.next_yield = 0

    def __next__(self):
        with self.lock:
            t = time.monotonic()
            if t < self.next_yield:
                time.sleep(self.next_yield - t)
                t = time.monotonic()
            self.next_yield = t + self.interval


class PerspectiveAPIMetric():
    def __init__(self):
        self._API_KEY = PERSPECTIVE_API_KEY
        self.attributes = ['TOXICITY', 'SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'INSULT', 'PROFANITY', 'THREAT']

    def compute(self, texts) -> List[float]:  
        # handle a signle or a list of examples
        def call_api(idx, text, retry=10):
            client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=self._API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
            try:
                analyze_request = {
                    'comment': {'text': text},
                    "languages": ["en"],
                    'requestedAttributes': {attribute:{} for attribute in self.attributes},
                }
                response = client.comments().analyze(body=analyze_request).execute()
                score = [response['attributeScores'][attribute]['summaryScore']['value'] for attribute in self.attributes]
                if len(score) != len(self.attributes):
                    score = [np.nan]*len(self.attributes)
                results.append((idx, score))
            except googleapiclient.errors.HttpError as e:
                if retry == 0:
                    print(e)
                    print('HTTPError, not retrying')
                    results.append((idx, [np.nan]*len(self.attributes)))
                else:
                    time.sleep(0.1)
                    print(f'HTTPError, retrying {retry-1}')
                    call_api(idx, text, retry=retry-1)

        def work_function(idx, text):
            next(api_rate_limiter)
            return call_api(idx, text)
        
        results = []
        api_rate_limiter = RateLimiter(1.5/QUOTA_IN_QPS)   # adjust rate based on quota, use formula: rate >= 1/(quota in QPS).
        
        chunk_size = 500
        for i_chunk in range(0, len(texts), chunk_size):
            gt = texts[i_chunk:i_chunk+chunk_size]

            with ThreadPoolExecutor(max_workers=40) as executor:
                for idx, text in enumerate(gt):
                    if text == "" or text is None:
                        print("text is None or empty String")
                        results.append((i_chunk+idx, [np.nan]*len(self.attributes)))
                    executor.submit(work_function, i_chunk+idx, text)

        scores = [np.nan] * len(texts)
        for idx, score in results:
            scores[idx] = score
        return scores