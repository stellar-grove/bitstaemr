import openai

def create_table_definition_prompt(df):
    """
    This function returns a prompt that informs GPT that we want to work with SQL Tables and what the overall goal is
    """

    prompt = '''Given the following sqlite SQL definition, write queries based on the request \n### sqlite SQL table, with its properties:
#
# Sales({})
#
'''.format(",".join(str(x) for x in df.columns))
    
    return prompt

def user_query_input():
    """Ask the user what they want to know about the data.

    Returns:
        string: User input
    """
    user_input = input("Tell OpenAi what you want to know about the data: ")
    return user_input


def send_to_openai(client, system_prompt, nlp_text):
    """Send the prompt to OpenAI

    Args:
        prompt (string): Prompt to send to OpenAI

    Returns:
        Response: Response from OpenAI
    """
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"A query to answer: {nlp_text}"},
      ]
    )
    return response
