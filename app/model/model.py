import pickle
import re
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]


# create predict_pipeline

def predict_pipeline(text):
    """
    Predicts the language of a given text.

    Parameters
    ----------
    text : str
        The text to be predicted.

    Returns
    -------
    str
        The predicted language.
    """

    
    # remove all symbol
    text=re.sub(r"!@#$%^&*","",text)
    # remove curl
    text=re.sub(r"[[]]","",text)
    # lower the text
    text=text.lower()
    #predict using model
    pred=model.predict([text])
    return classes[pred[0]]

