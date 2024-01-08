# Dataset based on #https://www.kaggle.com/datasets/kyanyoga/sample-sales-data

import os
import logging

import pandas as pd
from openai import OpenAI

import db_utils
import openai_utils

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == "__main__":
    client = OpenAI()

    logging.info("Loading data...")
    df = pd.read_csv("data/sales_data_sample.csv")
    logging.info(f"Data Format: {df.shape}")

    logging.info("Converting to database...")
    database = db_utils.dataframe_to_database(df, "Sales")
    
    system_prompt = openai_utils.create_table_definition_prompt(df)
    logging.info(f"Fixed SQL Prompt: {system_prompt}")

    logging.info("Waiting for user input...")
    user_input = openai_utils.user_query_input()

    logging.info("Sending to OpenAI...")
    response = openai_utils.send_to_openai(client, system_prompt, user_input)
    proposed_query = response.choices[0].message.content
    logging.info(f"Response obtained. Proposed sql query: {proposed_query}")
    result = db_utils.execute_query(database, proposed_query)
    logging.info(f"Result: {result}")
    print(result)