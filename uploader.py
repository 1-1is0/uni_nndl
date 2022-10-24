import requests

import os

token = "5290070754:AAGqk4Hgy0QdWBU9mr1jE-Vj4m5KKZ6sCQw"
chat_id = "1388536169"

tmp = list(os.scandir('.'))
for i in tmp:
    if 'tf_test' in i.name:
        file ={"document": open(f'{i.name}', 'rb')}
        res = requests.post(f"https://api.telegram.org/bot{token}/sendDocument?chat_id={chat_id}", files=file)
        print(res.content)
