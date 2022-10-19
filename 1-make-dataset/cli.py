from logic import get_reddits, reddits_to_df, get_tweets, tweets_to_df, get_subreddits
import click


@click.group()
def cli():
    pass


@cli.command()
@click.argument("output", type=click.File("wb"))
@click.option("--feed", default="hot")
@click.option("--limit", type=int, default=100)
def reddit(output, feed, limit):
    subreddits = get_subreddits(top_n=100)
    print(subreddits)
    reddits = []
    class_id = []
    for i,subr in enumerate(subreddits):
        reddit = [r for r in get_reddits(subr, feed, limit) if len(r.selftext)]
        print(reddit)
        reddits += reddit
        class_id.extend([i]*len(reddit))
    df = reddits_to_df(reddits)
    print(df)
    print(class_id)
    df['subreddit_id'] = class_id
    df.to_csv(output)


@cli.command()
@click.argument("output", type=click.File("wb"))
@click.argument("accounts", nargs=-1)
@click.option("--limit", type=int, default=100)
def twitter(output, accounts, limit):
    tweets = []
    for acc in accounts:
        tweets += get_tweets(acc, limit)
    df = tweets_to_df(tweets)
    df.to_csv(output, index=False)


if __name__ == "__main__":
    cli()
