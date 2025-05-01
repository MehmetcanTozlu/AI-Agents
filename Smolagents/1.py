import requests

money_1 = "USd".lower()
money_2 = "TRY".lower()
try:
    #print(dir(requests))
    url = f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/{money_1}.min.json"
    currency_request = requests.get(url).json()
    print(currency_request[money_1][money_2])
    print(currency_request['date'])
    print(f"Current date: {currency_request['date']}\nCurrent {money_1}/{money_2} rate: {currency_request[money_1][money_2]}")
except Exception as e:
    print(e)
