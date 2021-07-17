import json

## Load credentials from json file
cr = open("C:/Users/Admin/OneDrive - České vysoké učení technické v Praze/cred.json",)  ## change this path
data = json.load(cr)  

binance_api_key = data["binance_api_key"]        # API-key 
binance_api_secret = data["binance_api_secret"]  # API-secret
