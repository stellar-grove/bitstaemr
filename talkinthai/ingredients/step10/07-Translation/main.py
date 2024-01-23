import requests
import bs4
from openai import OpenAI

client = OpenAI()

# "Country" : (URL,HTML_TAG)
country_newspapers = {"Spain":('https://elpais.com/','.c_t'), 
                       "France":("https://www.lemonde.fr/",'.article__title-label')
                     }

def create_system_prompt():
    return "Detect the language of the news headlines below, then translate a summary of the headlines to English in a conversational tone."

def create_prompt():
    # Get Country
    country = input("What country would you like a news summary for? ")
    # Get country's URL newspaper and the HTML Tag for titles
    try:
        url,tag = country_newspapers[country]
    except:
        print("Sorry that country is not supported!")
        return
    
    # Scrape the Website
    results = requests.get(url)
    soup = bs4.BeautifulSoup(results.text,"lxml")
    
    # Grab all the text
    country_headlines = ''
    for item in soup.select(tag)[:3]:
        country_headlines += item.getText()+'\n'
        
    return country_headlines

def translate_and_summary(prompt):
    response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": create_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1, # Helps conversational tone a bit, optional
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=200,
    )
    print(response.choices[0].message)

if __name__ == "__main__":
    prompt = create_prompt()
    translate_and_summary(prompt)