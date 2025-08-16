## Equipade Knowledge Retrieval Benchmarks ##

In this repository we detail exactly what the Equipade Knowledge retrieval benchmark is, how the benchmark works, what it's meant to test, and what the rules for the benchmark are

# 1. What is the Equipade Knowledge Retrieval Benchmark

The benchmark was developed by Equipade to test how good a retrieval augmented generation system is at answering questions specific to industries pertaining vehicles and machines.

Throughout our experience doing RAG for these industries, we identified the following as pain points specific to them:
1. Traditional RAG systems are unable to properly handle previously unseen part numbers
2. Other RAG systems struggle with multi step reasoning tasks, where a RAG system needs to identify that a vehicle or machine number refers to a specific model. While LLMs have been trained to reason on their own, RAG is oftentimes still only 1-dimensional retrieval, and oftentimes that leads to situations where LLMs simply cannot figure out the right answer due to incomplete data
3. RAG systems need to understand the link between models and their member parts. Each model has specific parts, even though other parts might have similar names

In order to address these pain points, we first needed a way to test our solutions to see if we were making progress in addressing these pain points. The Equipade benchmarks were designed to do just that.


# How the benchmark works

The benchmark esentially asks a bunch of questions (which can be seen in the data/input_data.csv file) about a vehicle with a given VIN number. This VIN number can always be mapped to a vehicle model through the vin_table.csv. The BOM includes information on all components, and which models include those components. The questions asked are about components, so the RAG system will need to retrieve the correct information to understand what vehicle model we are working with, and what component we are talking about, since multiple components might have similar names. Identifying the correct model is essential for identifying the correct component and providing the right answer.

*diagram*

# Testing rules

In order to test a RAG system effectively, certain data ingestion and data retrieval rules were applied.

Specifically for data ingestion, we force all knowledge bases to take in the data from our bom.csv and vin_table.csv files as row by row text files. These text files are detailed in the /data/bom and /data/vin_table folders.

For retrieval, we restrict systems to only retrieve 5 rows of data from the knowledge base per question.

This is all to account for the fact that in real world applications, we will be dealing with much more data than would ever fit into 1 LLMs context window. Also, dumping in a whole company database for every query would be a terrible waste of tokens in any real-world application. So, we need to achieve good retrieval.

Realistically, the RAG systems here should never need more than 2-3 rows of data to get the right answer, assuming perfect retrieval, so we consider 5 rows of data a very reasonable amount of retrieved documents per question.

# Results

Currently, Equipade's backend achieves an accuracy of 95.70% when using gpt-4o as the LLM. An OpenAI RAG using File Search and gpt-4o achieves an accuracy of 70.66%. We hope to have metrics on other systems like google gemini with vertex ai soon.

# Extra information

If you are interested in Equipade's Voice and Text agent services, reach out to use at hello@equipade.com! You can also find more information about our services at equipade.com
