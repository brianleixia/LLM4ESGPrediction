#!/bin/bash

echo "start extract text from /data"
python extract_text.py

echo "start merge extracted text from /processed_data/extract_text"
python extract_text.py

echo "start keyword search process"
python keyword_search.py

