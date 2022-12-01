import cProfile
import sys
import math
import unidecode
import numpy as np
import typing
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class DocumentDistance:
    """
    Calculate the difference between documents, based
    on the TF-IDF values and the cosine differences.
    TF calculation methods:
        - "tf", Term frequency (default)
        - "raw", term count
        - "lognorm", log normalization
        - "doublenorm", double normalization
    IDF calculation methods:
        - "idf", Inverse Document Frequency (default)
        - "unary", Unary
        - "smooth", smoothed idf
        - "max", idf max
        - "prob, probablistic idf
        TFIDF calculation method (combination of TF and IDF methods):
        - "tfidf", TF IDF (default)
        - "doublenorm", doubble normalized TF and TF
        - "lognorm", log normalized TF and IDF
    """

    tf_method: str = "tf"
    idf_method: str = "idf"
    stopwords: set = {}
    stemmer: SnowballStemmer = None

    distances = None
    id2ref = []

    def __init__(self, tf_method="tf", idf_method="idf", tfidf=None, language="english"):
        self.distances = {}
        # Initialize methodology
        if tfidf:
            if tfidf == "doublenorm":
                self.tf_method = "doublenorm"
                self.idf_method = "idf"
            elif tfidf == "lognorm":
                self.tf_method = "lognorm"
                self.idf_method = "idf"
            elif tfidf == "tfidf":
                self.tf_method = "tf"
                self.idf_method = "idf"
            else:
                raise Exception("Invalid tfidf-parameter, valid options: tfidf, lognorm, doublenorm")
        else:
            if tf_method in ['tf', 'raw', 'lognorm', 'doublenorm']:
                self.tf_method = tf_method
            else:
                raise Exception("Invalid tf-parameter, valid options: tf, raw, lognorm, doublenorm")
            if idf_method in ['idf', 'unary', 'smooth', 'max', 'prob']:
                self.idf_method = idf_method
            else:
                raise Exception("Invalid idf-parameter, valid options: idf, unary, smooth, max, prob")
        # Stopwords and stemmer
        avail_lang = set(stopwords.fileids()).intersection(SnowballStemmer.languages)
        if not language in avail_lang:
            raise Exception("Invalid language, valid options: " + str(avail_lang))
        self.stopwords = set(stopwords.words(language))
        self.stemmer = SnowballStemmer(language)

    def clean_text(self, doc: str) -> str:
        """
        Clean a text before tokenization.
        Convert to lower case, unidecoded, and accents removed
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
        - Stopwords are removed
        - Short words (length < 3) are removed
        - Numerical words are removed
        - Stemming is applied to the words
        :param words: Original wordlist
        :return: Cleaned wordlist
        """
        words = [x for x in words if x not in self.stopwords and len(x) > 2 and not x.isnumeric()]
        words = [self.stemmer.stem(plural) for plural in words]
        return words

    def pre_proces_data(self,
                        documents: typing.List[str]
                        ) -> (typing.Dict[int, typing.Dict[str, int]], typing.Dict[int, int], typing.Dict[str, int]):
        """
        Pre proces the documents
        Translate a dictionary of ID's and sentences to two dictionaries:
        - bag_of_words: dictionary with IDs and a list with the words in the text
        - word_occurences: dictionary with IDs and per document word counts
        :param documents:
        :return: dictionary with word count per document, dictionary with sentence lengths
        """
        # Tokenize sentences and determine the complete set of unique words
        bag_of_words = []
        unique_words = set()
        i = 0
        for doc in documents:
            i += 1
            doc = self.clean_text(doc)
            words = word_tokenize(doc)
            words = self.clean_word_list(words)
            bag_of_words.append(words)
            unique_words = unique_words.union(set(words))
        # Determine word occurences in each sentence for all words
        word_occurences = []
        sentence_lengths = []
        for words in bag_of_words:
            now = dict.fromkeys(unique_words, 0)
            for word in words:
                now[word] += 1
            word_occurences.append(now)
            sentence_lengths.append(len(words))

        # Now clean op the lists
        # Fill dictionary with '0' for each word
        doc_count_per_word = dict.fromkeys(word_occurences[0], 0)
        # Travese all documents and words
        # If a word is present in a document, the doc_count_per_word value of
        # the word is increased
        for document in word_occurences:
            for word, val in document.items():
                if val > 0:
                    doc_count_per_word[word] += 1
        # # Get all words that occur in one document and thus have no value
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

    def compute_tf(self,
                   wordcount: typing.Dict[str, int],
                   doc_length: int
                   ) -> typing.Dict[str, float]:
        """
        Calculates the Term Frequency (TF)
        This is the number of times a word appears in the document divided by
        the total number of words in the document
        :param wordcount: dictionary with mapping from word to count
        :param doc_length: list of words in the sentence
        :return: dictionary mapping word to its frequency
        """
        tf_dict = {}
        if self.tf_method == "raw":
            for word, count in wordcount.items():
                tf_dict[word] = count
        elif self.tf_method == "lognorm":
            for word, count in wordcount.items():
                tf_dict[word] = math.log(1 + count)
        elif self.tf_method == "doublenorm":
            maxcount = max((list(wordcount.values())))
            for word, count in wordcount.items():
                tf_dict[word] = 0.5 + 0.5 * (count / maxcount)
        else:  # method TF or invalid specification
            for word, count in wordcount.items():
                tf_dict[word] = count / float(doc_length)
        return tf_dict

    def compute_idf(self,
                    doc_count_per_word: typing.List[typing.Dict[str, int]],
                    no_documents: int
                    ) -> typing.Dict[str, int]:
        """
        Calculates the inverse data frequency (IDF)
        The smaller the IDF, the more frequent the word occurs. If a word
        appears in all documents, the IDF is 0.
        :param doc_count_per_word: dictionary with all documents. A document is a dictionary of TF
        :param no_documents: number of documents
        :return: IDF value for all words
        """
        # doc_count_per_word contains for each word the number of documents
        # it occurs in. Calculte the IDF from this data
        if self.idf_method == "unary":
            for word, val in doc_count_per_word.items():
                doc_count_per_word[word] = 1.0
        elif self.idf_method == "smooth":
            for word, val in doc_count_per_word.items():
                doc_count_per_word[word] = 1.0 + math.log(no_documents / (1.0 + float(val)))
        elif self.idf_method == "prob":
            for word, val in doc_count_per_word.items():
                doc_count_per_word[word] = math.log((no_documents - val + 1) / float(val))
        elif self.idf_method == "max":
            maxval = max((list(doc_count_per_word.values())))
            for word, val in doc_count_per_word.items():
                doc_count_per_word[word] = math.log(maxval / (1 + float(val)))
        else:  # idf and default
            for word, val in doc_count_per_word.items():
                doc_count_per_word[word] = math.log(no_documents / float(val))
        return doc_count_per_word

    def compute_tfidf(self,
                      word_counts: typing.Dict[str, float],
                      idfs: typing.Dict[str, float]
                      ) -> typing.Dict[str, float]:
        """
        Calculte the TF-IDF score for all words for a document
        THis is the TF multiplied with the IDF value
        :param word_counts: Dictionary mapping words to their number of occurences in the sentence
        :param idfs: Dictionary with the IDF value for all words
        :return: TF-IDF values for all words
        """
        tfidf = {}
        for word, val in word_counts.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    def normalize(self,
                  tfidfs: typing.List[typing.Dict[str, float]]
                  ) -> typing.Dict[int, float]:
        """
        Normalize the dictionary entries (first level)
        E.g. {A: {a: 3, b: 4}} returns {A: 5} where 5 is the normalized value for [3. 4]
        :param tfidfs: dictiory of dictionarys
        :return: dictionary with normalized values
        """
        norms = []
        for i in range(len(tfidfs)):
            vals = list(tfidfs[i].values())
            sumsq = 0
            for j in range(len(vals)):
                sumsq += pow(vals[j], 2)
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
        tfidfs_opt = []
        # Extract arrays of numbers
        for tfidf in tfidfs:
            tfidfs_opt.append(list(tfidf.values()))

        self.distances = [[1.0] * len(tfidfs_opt) for _ in range(len(tfidfs_opt))]
        for key_1 in range(len(tfidfs_opt)):
            for key_2 in range(key_1 + 1, len(tfidfs_opt)):
                distance = np.dot(tfidfs_opt[key_1], tfidfs_opt[key_2]) / (norms[key_1] * norms[key_2])
                distance = round(distance, 2)
                self.distances[key_1][key_2] = distance
                self.distances[key_2][key_1] = distance

    def add_documents(self,
                      documents: typing.Dict[int, str],
                      ) -> None:
        """
        Calculate the distance between the documents
        :param documents: dictionary with IDs and documents
        :return: None
        """

        # Store references and restructure to an array
        self.id2ref = list(documents.keys())
        docs = list(documents.values())

        word_occurence_per_doc, doc_lengths, doc_count_per_word = self.pre_proces_data(docs)

        # Calculate TF values
        tfs = []
        for i in range(len(word_occurence_per_doc)):
            tfs.append(self.compute_tf(word_occurence_per_doc[i], doc_lengths[i]))

        # Calculate IDF values
        idfs = self.compute_idf(doc_count_per_word, len(docs))

        # Calculate TF-IDF values
        tfidfs = []
        for i in range(len(tfs)):
            tfidfs.append(self.compute_tfidf(tfs[i], idfs))

        # Calculate distances
        self.calculate_distances(tfidfs)

    def get_distances(self, docid: int) -> typing.Dict[int, int]:
        # Find index and get distances
        idx = self.id2ref.index(docid)
        res = dict(zip(self.id2ref, self.distances[idx]))
        # sort distances (descending)
        res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
        return res

    
def main() -> int:
    distance = DocumentDistance()
    distance.add_documents({10: 'the yellow man went out for a walk',
                            11: 'the children sat around the yellow fire',
                            22: 'the children walk around the yellow house'})
    print(distance.distances)


if __name__ == '__main__':
    cProfile.run('main()', 'prof_opt.out')
