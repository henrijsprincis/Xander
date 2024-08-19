import time
from openai import OpenAI
client = OpenAI()

def ask_chatGPT(query, model = "gpt-4o-mini"):
    try:
        completion = client.chat.completions.create(
            model= model,
            messages=[{"role": "user", "content": query}],
            temperature=0,
            top_p=0.01,
        )
        return(completion.choices[0].message.content)
    except Exception as e:
        print("Error occured: Retrying in 30 seconds. Error: ", e)
        time.sleep(30)
        return ask_chatGPT(query)