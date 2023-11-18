import re

from autocorrect import Speller
from nltk.tokenize import word_tokenize

spell = Speller(lang='en')


def reduce_lengthening(text: str) -> str:
    """
    Reduce consecutive character lengthening in the input text.

    This function identifies consecutive character lengthening (e.g., "sooo" or "coooool") in the text and replaces
    them with double characters (e.g., "soo" or "cool").

    Parameters:
        text (str): The input text containing potential character lengthening.

    Returns:
        str: The text with consecutive character lengthening reduced.

    Example:
        # >>> reduced_text = reduce_lengthening("sooo good!")
        # >>> print(reduced_text)
        "soo good!"
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)


def text_preprocess(doc: str) -> str:
    """
    Perform text preprocessing on the input document.

    This function performs several text preprocessing steps, including lowercase conversion, removal of hashtags,
    mentions, links, numbers, and more.
    It also tokenizes the text, reduces word lengthening, corrects spelling

    Parameters:
        doc (str): The input document to be preprocessed.

    Returns:
        str: The preprocessed text.

    Example:
        # >>> preprocessed_text = text_preprocess("I love this product! It's amazing!!!")
        # >>> print(preprocessed_text)
        "love product amazing"
    """
    # Lowercase all the letters
    temp = doc.lower()
    # Removing punctuation
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)
    temp = re.sub("#[A-Za-z0-9_]+", "", temp)
    temp = re.sub(r'[^\w\s]', '', temp)
    # Removing links
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub(r"www.\S+", "", temp)
    # removing numbers
    temp = re.sub("[0-9]", "", temp)

    # Tokenization
    temp = word_tokenize(temp)
    # Fixing Word Lengthening
    temp = [reduce_lengthening(w) for w in temp]
    # spell corrector
    temp = [spell(w) for w in temp]

    temp = " ".join(w for w in temp)

    return temp

# if __name__ == '__main__':
#     print(text_preprocess("I love this product! It's amazing!!!"))
