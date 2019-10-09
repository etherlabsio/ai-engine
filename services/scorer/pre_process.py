from typing import List
import pickle
import os
import text_preprocessing.preprocess as tp

def preprocess_text(text: str) -> List[str]:
    mod_texts_unfiltered = tp.preprocess(text, stop_words=False, remove_punct=False)
    mod_texts = []
    if len(mod_texts_unfiltered) > 1:
        for sentence in mod_texts_unfiltered:
            if len(sentence.split(' ')) > 3:
                mod_texts.append(sentence)
    return mod_texts
