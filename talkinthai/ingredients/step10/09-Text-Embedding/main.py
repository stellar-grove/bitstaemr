from ast import literal_eval

from openai import OpenAI
import numpy as np
import pandas as pd


client = OpenAI()

df = pd.read_csv('unicorns_with_embeddings.csv')
df["embedding"] = df["embedding"].apply(literal_eval)

def get_embedding(text):
    result = client.embeddings.create(
      model='text-embedding-ada-002',
      input=text
    )
    return result.data[0].embedding


def vector_similarity(vec1,vec2):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(vec1), np.array(vec2))

def embed_prompt_lookup():
    # initial question
    question = input("What question do you have about a Unicorn company? ")
    # Get embedding
    prompt_embedding = get_embedding(question)
    # Get prompt similarity with embeddings
    # Note how this will overwrite the prompt similarity column each time!
    df["prompt_similarity"] = df['embedding'].apply(lambda vector: vector_similarity(vector, prompt_embedding))

    # get most similar summary
    summary = df.nlargest(1,'prompt_similarity').iloc[0]['summary'] 

    prompt = f"""Only answer the question below if you have 100% certainty of the facts, use the context below to answer.
            Here is some context:
            {summary}
            Q: {question}
            A:"""


    response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=500,

    )
    print(response.choices[0].message.content.strip(" \n"))

if __name__ == "__main__":
    embed_prompt_lookup()