from typing import List
import pickle
import os
import text_preprocessing.preprocess as tp
import re


def matcher(matchObj):
    return (
        matchObj.group(0)[0]
        + matchObj.group(0)[1]
        + " "
        + matchObj.group(0)[2]
    )


def preprocess_text(text: str) -> List[str]:
    text = re.sub("[a-z][.?][A-Z]", matcher, text)
    mod_texts_unfiltered = tp.preprocess(
        text, stop_words=False, remove_punct=False
    )
    mod_texts = []
    if len(mod_texts_unfiltered) >= 1:
        for sentence in mod_texts_unfiltered:
            if len(sentence.split(" ")) > 0:
                if sentence[-1] not in [".", "?", "!"]:
                    sentence += "."
                sentence = sentence.replace("?.", "?")
                mod_texts.append(sentence)
    return mod_texts
