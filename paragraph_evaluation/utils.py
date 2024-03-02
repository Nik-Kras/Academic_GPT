from openai import OpenAI
import os

API_KEY = os.environ['API_KEY']

def model_ask(paragraph, prompt):

  client = OpenAI(api_key=API_KEY)

  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
      {"role": "system", "content": prompt},
      {"role": "user", "content": paragraph}
    ],
    response_format={ "type": "json_object" }
  )

  return completion.choices[0].message
