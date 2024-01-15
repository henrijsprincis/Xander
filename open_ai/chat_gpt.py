import time
import openai
import os


def ask_chatGPT(query):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Please define the OPENAI_API_KEY variable to use chatgpt")
        exit()
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                messages=[{"role": "user", "content": query}],
                                                temperature = 0,
                                                top_p = 0.01
                                                )
        return(completion.choices[0].message.content)
    except:
        print("Error occured: Retrying in 30 seconds")
        time.sleep(30)
        return ask_chatGPT(query)