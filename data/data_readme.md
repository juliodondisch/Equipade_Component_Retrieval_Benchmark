The data in this folder is what is used for the benchmarks.

The csv tables vin_table.csv and bom.csv are the data that is fed into a knowledge base and which the tested systems can use to generate answers. 

It is important to note that the data is put into the RAG systems in row-by-row text format, as shown in the bom and vin_table folders, not as the whole csv together. This is because the goal of this benchmark is to test RETRIEVAL ability. In real-world applications, we would expect much more tokens than would fit in the context window for 1 model, so we limit each RAG system to only pull up to 5 retrieved rows out at a time to generate an answer.

The input_data.csv and answer_key.csv files are our validation data. For each input question in answer_key.csv, we expect the answer from a tested RAG system to include the correct answer. We evaluate answers with an LLM-as-a-judge system which is only exposed to the real answer and expected answer. The same LLM as a judge system is used for all benchmarks, with the system prompt for this llm as a judge detailed in the testing scripts.