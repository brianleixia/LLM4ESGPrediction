cat env.txt soc.txt gov.txt | sort | uniq > corpus.txt
cat gov/*.jsonl > gov/combined.jsonl
cat env/*.jsonl > env/combined.jsonl
cat soc/*.jsonl > soc/combined.jsonl
cat nonesg_sft_data_wtext.jsonl esg_classification_sft.jsonl > esg_classification_sft_data.jsonl
