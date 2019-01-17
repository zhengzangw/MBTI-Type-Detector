import tweepy
auth = tweepy.OAuthHandler('your API Key', 'your API Secret')
auth.set_access_token('your Access Token', 'your Access Token Secret')
api = tweepy.API(auth)