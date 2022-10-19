from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd

from logic import get_reddits, reddits_to_df, get_tweets, tweets_to_df, get_subreddits


app = FastAPI()


class RedditRequest(BaseModel):
    dest: str


@app.post("/reddit")
def api_reddit(req: RedditRequest):
    try:
        df = pd.read_csv("output.csv", index_col=None).fillna(value = "None")
        print(df)
        df.to_gbq(req.dest, if_exists="replace") #write to the Big Query bucket specified by dest. change if you want some other behavior.
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
