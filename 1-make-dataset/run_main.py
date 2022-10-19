from typing import List
from pydantic import BaseModel
from fastapi import FastAPI

from logic import get_reddits, reddits_to_df, get_tweets, tweets_to_df, get_subreddits


app = FastAPI()


class RedditRequest(BaseModel):
    feed: str = "hot"
    post_limit: int = 100
    subreddit_limit: int = 100
    dest: str


@app.post("/reddit")
def api_reddit(req: RedditRequest):
    try:
        reddits = []
        class_id = []
        subreddits = get_subreddits(top_n=req.subreddit_limit)
        for i,subr in enumerate(subreddits):
            reddit = [r for r in get_reddits(subr, req.feed, req.post_limit) if len(r.selftext)]
            reddits += reddit
            class_id.extend([i]*len(reddit))
        df = reddits_to_df(reddits)
        df['subreddit_id'] = class_id
        df.to_gbq(req.dest) #write to the Big Query bucket specified by dest. change if you want some other behavior.
    except Exception as e:
        return repr(e)
    return "OK"


class TwitterRequest(BaseModel):
    accounts: List[str]
    limit: int = 100
    dest: str


@app.post("/twitter")
def api_twitter(req: TwitterRequest):
    try:
        tweets = []
        for acc in req.accounts:
            tweets += get_tweets(acc, req.limit)
        df = tweets_to_df(tweets)
        df.to_gbq(req.dest)
    except Exception as e:
        return repr(e)
    return "OK"
