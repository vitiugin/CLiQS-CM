# CLiQS-CM
Cross-Lingual Query-Based Summarization of Crisis-Related Social Media:
An Abstractive Approach Using Transformers



## Data

Folder **data** includes **tweets** collected during five events and **reports** provided by ERCC. Every file in folder **tweets** (inside event directories) contains columns with tweet *ids*, *labels* (informative crisis [1] /non-crisis [0] and in some files non-informative crisis [2]), and *categories* ([1] - tweet labeled as category related). Folders inside **reports** include crisis-related .pdf files downloaded from [ERCC Echo Flash](https://erccportal.jrc.ec.europa.eu/ECHO-Products/Echo-Flash#/echo-flash-items/latest) and text reports used as ground truth during summaries evaluation.

Additionally, folder **data** includes file **queries.json** with queries for every dataset and category.



## Features extraction

For extraction features needed for reproducing the model, you could use file **features_extraction.py** with arguments *datafile*, *dataset name*, *category*, and *language* (ISO 639-1 code).

```python
python3 features_extraction.py 'taal_out_en.csv' 'taal' 'Danger' 'en'
```

The output files with text features, similarity features, and sentence embeddings vectors will be stored in the same directory.



## Classification

For training and evaluating the classification model, you could execute classification.py script with the arguments *dataset name* and *test language* (ISO 639-1 code of the test language. The algorithm will train on all other languages allowed for the event). 

```python
python3 classification.py 'australia' 'en'
```



Also, it is necessary to have three files in the same directory for every language in the dataset prepared using feature_extraction.py. For Taal volcano eruption set in English it would be next files: *taal_en.csv*, *taal_en_laser_features.csv*, *taal_en_text_features.to_csv*.



## Instllation

Python 3.7.2
