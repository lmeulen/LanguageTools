import typing
import unidecode
import math
import numpy as np
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords


class DocumentDistance:
    distances = {}

    stop_words = set(stopwords.words("dutch"))
    stemmer = SnowballStemmer("dutch")

    def add_documents(self, docs: typing.List[str],
                      ) -> None:
        """
        Calculate the distance between the documents
        :param documents: list of documents
        :return: None
        """
        print('Documents:')
        print(docs)
        print()
        word_occ_per_doc, doc_lens, doc_per_word = self.pre_proces_data(docs)
        print('word_occ_per_doc:')
        print(word_occ_per_doc)
        print()
        print('doc_lens:')
        print(doc_lens)
        print()
        print('doc_per_word:')
        print(doc_per_word)
        print()
        
        # Calculate TF values
        tfs = []
        for i in range(len(word_occ_per_doc)):
            tfs.append(self.compute_tf(word_occ_per_doc[i], doc_lens[i]))
        print('tfs:')
        print(tfs)
        print()

        # Calculate IDF values
        idfs = self.compute_idf(doc_per_word, len(docs))
        print('idfs:')
        print(idfs)
        print()

        # Calculate TF-IDF values
        tfidfs = []
        for i in range(len(tfs)):
            tfidfs.append(self.compute_tfidf(tfs[i], idfs))
        print('tfidfs:')
        print(tfidfs)
        print()

        # Calculate distances
        self.calculate_distances(tfidfs)
        
    def pre_proces_data(self,
                        documents: typing.List[str]
                        ) -> (typing.Dict[int, typing.Dict[str, int]], 
                             typing.Dict[int, int], 
                             typing.Dict[str, int]):
        """
        Pre proces the documents
        Translate a dictionary of ID's and sentences to two dictionaries:
        - bag_of_words: dictionary with IDs and a list with the words in the text
        - word_occurences: dictionary with IDs and per document word counts
        :param documents:
        :return: dictionary with word count per document, dictionary with sentence lengths
        """
        # 1. Tokenize sentences and determine the complete set of unique words
        bag_of_words = []
        unique_words = set()
        for doc in documents:
            doc = self.clean_text(doc)
            words = word_tokenize(doc)
            words = self.clean_word_list(words)
            bag_of_words.append(words)
            unique_words = unique_words.union(set(words))
        # 2. Determine word occurences in each sentence for all words
        word_occurences = []
        sentence_lengths = []
        for words in bag_of_words:
            now = dict.fromkeys(unique_words, 0)
            for word in words:
                now[word] += 1
            word_occurences.append(now)
            sentence_lengths.append(len(words))

        # 3. Count documents per word
        doc_count_per_word = dict.fromkeys(word_occurences[0], 0)
        # Travese all documents and words
        # If a word is present in a document, the doc_count_per_word value of
        # the word is increased
        print(word_occurences)
        for document in word_occurences:
            for word, val in document.items():
                if val > 0:
                    doc_count_per_word[word] += 1

        # 4. Drop words appearing in one or all document
        words_to_drop = []
        no_docs = int(len(documents) * .90)
        for word, cnt in doc_count_per_word.items():
            if cnt == 1 or cnt > no_docs:
                words_to_drop.append(word)
        for word in words_to_drop:
            doc_count_per_word.pop(word)
            for sent in word_occurences:
                sent.pop(word)
                    
        return word_occurences, sentence_lengths, doc_count_per_word
    
    def clean_text(self, doc: str) -> str:
        """
        Clean a text before tokenization.
        :param doc: The text to clean
        :return: cleaned text
        """
        doc = doc.lower()
        doc = unidecode.unidecode(doc)
        doc = doc.replace('_', ' ').replace("'", "")
        return doc
    
    def clean_word_list(self, words: typing.List[str]) -> typing.List[str]:
        """
        Clean a worlist
        :param words: Original wordlist
        :return: Cleaned wordlist
        """
        words = [x for x in words if x not in self.stop_words and len(x) > 2 and not x.isnumeric()]
        words = [self.stemmer.stem(plural) for plural in words]
        return list(set(words))
    
    def compute_tf(self,
                   wordcount: typing.Dict[str, int],
                   no_words: int
                   ) -> typing.Dict[str, float]:
        """
        Calculates the Term Frequency (TF)
        :param wordcount: dictionary with mapping from word to count
        :param no_words: word count of the sentence
        :return: dictionary mapping word to its frequency
        """
        tf_dict = {}
        for word, count in wordcount.items():
            tf_dict[word] = float(count) / no_words
        return tf_dict

    def compute_idf(self,
                    doc_count_per_word: typing.List[typing.Dict[str, int]],
                    no_documents: int
                    ) -> typing.Dict[str, int]:
        """
        Calculates the inverse data frequency (IDF)
        :param doc_count_per_word: dictionary with all documents. A document is a dictionary of TF
        :param no_documents: number of documents
        :return: IDF value for all words
        """
        idf_dict = {}
        for word, val in doc_count_per_word.items():
            idf_dict[word] = math.log(float(no_documents) / val)
        return idf_dict

    def compute_tfidf(self,
                       tfs: typing.Dict[str, float],
                       idfs: typing.Dict[str, float]
                       ) -> typing.Dict[str, float]:
        """
        Calculte the TF-IDF score for all words for a document
        :param tfs: TFS value per word
        :param idfs: Dictionary with the IDF value for all words
        :return: TF-IDF values for all words
        """
        tfidf = {}
        for word, val in tfs.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    def normalize(self,
                  tfidfs: typing.List[typing.Dict[str, float]]
                  ) -> typing.Dict[int, float]:
        """
        Normalize the dictionary entries (first level)
        :param tfidfs: dictiory of dictionarys
        :return: dictionary with normalized values
        """
        norms = []
        for i in range(len(tfidfs)):
            vals = list(tfidfs[i].values())
            sumsq = 0
            for i in range(len(vals)):
                sumsq += pow(vals[i], 2)
            norms.append(math.sqrt(sumsq))
        return norms

    def calculate_distances(self,
                            tfidfs: typing.List[typing.Dict[str, float]]
                            ) -> None:
        """
        Calculate the distances between all elements in tfidfs
        :param tfidfs: The dictionary of dictionaries
        :return: None
        """
        norms = self.normalize(tfidfs)
        vectors = []
        # Extract arrays of numbers
        for tfidf in tfidfs:
            vectors.append(list(tfidf.values()))

        self.distances = [[1.0] * len(vectors) for _ in range(len(vectors))]
        for key_1 in range(len(vectors)):
            for key_2 in range(key_1 + 1, len(vectors)):
                distance = np.dot(vectors[key_1], vectors[key_2]) / (norms[key_1] * norms[key_2])
                self.distances[key_1][key_2] = distance
                self.distances[key_2][key_1] = distance
        print()
        print(self.distances)
        
distance = DocumentDistance()
distance.add_documents(['the man walked around the green house',
                        'the children sat around the fire',
                        'a man set a green house on fire'])
