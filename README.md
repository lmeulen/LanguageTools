# LanguageTools

This package contains several language tools. 

## Document distance

A class to determine the distance between different documents in a corpus.

Related articles
- [Finding related articles with TF-IDF and Python](https://towardsdatascience.com/finding-related-articles-with-tf-idf-and-python-d6e1cd10f735?sk=4e7185f4de845392e99035d8c22751a5)

Implementation
  - NaiveDistance.py : Implementaiton as discussed in my TDS article
  - Distance.py: More extended implementation, including different calculations of TF, IDF and TFIDF
  
## Summarize text

Class to summarize a long text, supports multiple languages.

Related articles 
- [Summarize a text in Python](https://towardsdatascience.com/summarize-a-text-with-python-b3b260c60e72?sk=9d66f3557b7f41b4e7eae1688c5b8120)
- [Summarize a text in Python - continued](https://towardsdatascience.com/summarize-a-text-with-python-b3b260c60e72?sk=9d66f3557b7f41b4e7eae1688c5b8120)

Implementation
- SummarizeText.py - Python class implementing the functionality

## Parse Fixed Width File

Parse a fixed width file to a list of dictionaries or a Pandas dataframe.

Related article
- [Parsing fixed width text files with Python]()

Implementation
- Parse_Fixed_Width_File.py
