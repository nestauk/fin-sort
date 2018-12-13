from collections import Counter
from docx import Document
import numpy as np
import os
from PyPDF2 import PdfFileReader
from PyPDF2.utils import PdfReadError
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine

"""Local directory containing project proposals"""
DOC_BASE_DIR = "/Users/jklinger/Downloads/Proposals/"


def iterdir(path, files_only=False, dirs_only=False):
    """Iterate over a file path, rejoining any items with their path

    Args:
        path (str): Relative path to iterate through.
        files_only (bool): Only consider file items in :obj:`path`.
        dirs_only (bool): Only consider dir items in :obj:`path`.
    Yields:
        Full path to item in :obj:`path`.
    """
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if files_only and not os.path.isfile(item_path):
            continue
        if dirs_only and not os.path.isdir(item_path):
            continue
        yield item_path


class Reader(list):
    """Data structure for scraped (SQL) data and for data from PDFs
    and docx files.

    Args:
        sql_url (str): SQL connection URL.
        table_name (str): Table name in SQL DB.
        field_name (str): Field name in SQL table.
    """
    def __init__(self, sql_url, table_name, field_name):
        self._labels = []  # For keeping tabs on labels

        # Read data from the database
        engine = create_engine(sql_url)
        df = pd.read_sql(table_name, engine, columns=[field_name])
        for text in df.values:
            self.append('\n'.join(text))
            self._labels.append(0)

        # Read data from files
        self._read = {"pdf": self.read_pdf,
                      "docx": self.read_docx}

    def labels(self):
        """Convert internal labelling into boolean np array."""
        return np.array(self._labels) == 1

    def read(self, fname, label):
        """Wrapper read method, for PDF or docx files.

        Args:
            fname (str): Filename to read.
            label (int): Integer label for this data.
        """
        suffix = fname.split(".")[-1].lower()
        text = self._read[suffix](fname)
        self.append("\n".join(t for t in text if len(t) > 0))
        self._labels.append(label)

    @staticmethod
    def read_pdf(fname):
        try:
            pdf = PdfFileReader(fname)
        except PdfReadError:
            return ""
        else:
            text = [p.extractText() for p in pdf.pages]
        return text

    @staticmethod
    def read_docx(fname):
        docx = Document(fname)
        text = [p.text for p in docx.paragraphs]
        return text


if __name__ == "__main__":

    # Read the background data
    reader = Reader("sqlite:///data/data.db", "fin_unfiltereddata", "content")
    # Read the signal data
    for dir_path in iterdir(DOC_BASE_DIR, dirs_only=True):
        for file_path in iterdir(dir_path, files_only=True):
            text = reader.read(file_path, label=1)

    # Binarise term counts
    cv = CountVectorizer(strip_accents='ascii',
                         lowercase=True,
                         stop_words="english",
                         ngram_range=(1, 2),
                         max_df=0.75,
                         min_df=0,
                         max_features=10000,
                         binary=True)
    data = cv.fit_transform(reader)
    vocab_len = data.shape[1]

    # Extract signal and background vectors
    signal = data[reader.labels()]
    background = data[~reader.labels()]

    # Exclude any vocab that doesn't appear in both datasets
    overlapping_vocab = (signal.sum(axis=0) > 0) & (background.sum(axis=0) > 0)
    overlapping_vocab = np.array(overlapping_vocab).reshape((vocab_len,))
    overlapping_vocab = np.tile(overlapping_vocab, reps=(data.shape[0], 1))
    _data = sparse.csr_matrix(data.multiply(overlapping_vocab))
    signal = _data[reader.labels()]
    background = _data[~reader.labels()]

    # Parameters for training
    bkg_idxs = range(background.shape[0])  # indexes for random strap selection
    strapsize = signal.shape[0]  # Set strap size to the length of the signal
    n_straps = 100  # Hyperparameter: how many random straps to sample
    _labels = pd.Series([0]*strapsize + [1]*strapsize)  # Dummy strap labels

    # The output vector: sum of correlation coefficients
    feature_importance = np.zeros(vocab_len)

    # Iterate over straps
    for strap in range(0, n_straps):
        # Generate the strap sample
        idxs = np.random.choice(bkg_idxs, strapsize)
        bkg = background[idxs]
        _data = sparse.vstack((bkg, signal))
        # Calculate the correlation coefficient
        _df = pd.DataFrame(_data.toarray())
        corr = _df.apply(lambda x: x.corr(_labels))
        # Ignore small or NaN values, they're not interesting
        corr.loc[pd.isnull(corr)] = 0
        corr.loc[np.abs(corr) < 1e-4] = 0
        # Incremeent the output vector
        feature_importance += corr

    # Word : weight mapping
    results = {k: feature_importance[v]
               for k, v in cv.vocabulary_.items()}
    results = Counter(results)
    df = pd.DataFrame(Counter(results).most_common(),
                      columns=["word", "weight"])
    df.to_csv("data/keywords.csv", index=False)
