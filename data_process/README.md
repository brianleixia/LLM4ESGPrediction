# Data Processing Workflow

This document outlines the specific steps involved in the data processing pipeline, including the sources of our data, how we extract relevant information through **Keyword Search**, and **Using LLMs' APIs for data labeling**.

## Data Sources

The table below details the datasets used in this project, download them and put them under folder **/data**, along with their descriptions, sizes, types, and class information:

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Description</th>
            <th>Size</th>
            <th>Type</th>
            <th>Classes</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>
<a href="https://huggingface.co/datasets/Abhijeet3922/ESG-Prospectus-Clarity-Category" rel="noopener" target="_blank">ESG-Prospectus-Clarity-Category</a></td>
            <td>&lt;text, label&gt;</td>
            <td>1160 rows</td>
            <td>Classification: 4 classes</td>
            <td>Specific; Ambiguous; Generic; Risk</td>
        </tr>
        <tr>
            <td>
<a href="https://huggingface.co/datasets/TrajanovRisto/esg-sentiment" rel="noopener" target="_blank">esg-sentiment</a></td>
            <td>&lt;text, label 1, …, label 9&gt;</td>
            <td>679 rows</td>
            <td>Classification: 9 classes</td>
            <td>&lt;Environmental, Social, Governance&gt; * &lt;Positive, Neutral, Negative&gt;</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/ESGBERT" rel="noopener" target="_blank">ESGBERT</a></td>
            <td>&lt;sentence&gt;</td>
            <td>Environmental_2k: 2000 rows
Social_2k:2000 rows
Governance_2k: 2000 rows
Action_500: 500 rows
Total: 6500 rows</td>
            <td>Classification</td>
            <td>Split data into: action, environmental, social, governance. Each labeled: &lt;0, 1&gt;</td>
        </tr>
        <tr>
            <td> <a href="https://www.kaggle.com/datasets/equintel/dax-esg-media-dataset?select=esg_documents_for_dax_companies.csv" rel="noopener" target="_blank">DAX ESG Media Dataset</a></td>
            <td>-</td>
            <td>11455 rows</td>
            <td>Classification</td>
            <td>-</td>
        </tr>
        <tr>
            <td>
<a href="https://huggingface.co/datasets/climatebert/environmental_claims?row=95">Environmental_claims</a>
</td>
            <td>-</td>
            <td>272 kB</td>
            <td>Classification</td>
            <td>environmental claim: Yes / No</td>
        </tr>
        <tr>
            <td>
<a href="https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html">CLIMATE-FEVER</a></td>
            <td>-</td>
            <td>3MB</td>
            <td>1,535 real-world claims regarding climate-change. 7,675 claim-evidence pairs.
            <td>-</td>

</td>
        </tr>
    </tbody>
</table>

## Keyword Search

The first step in extracting relevant data involves a keyword search. We define a series of keywords associated with ESG and search for relevant content within the datasets mentioned above. This step helps us focus on specific issues and discussions within the ESG domain. Run this scripts to perform data preprocessing and keyword search

```bash
./run_keyword_search.sh
```

## Labeling with LLMs' APIs

To further refine the data and enhance its utility, we employ Language Models (such as GPT-3 or BERT) for automated data labeling. This step involves:

- Preprocessing the extracted text, including cleaning and normalization.
- Using LLMs' APIs to classify the text, which contains three tasks. Code refer folders: task1, task2, task3
- Construct the pretaining dataset, classificatino labeled dataset and supervised finetunting (SFT) dataset.、

To run this, you should run the following scripts step by step according to each task. For example, you want to use ChatGPT to do data labeling on Environmental domain:

```bash
./run_data_select_gpt.sh
```

```bash
./run_classify_gpt_env.sh
```

```bash
./run_9classify_gpt_env.sh
```
