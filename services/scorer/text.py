from typing import List


def append_punctuation_at_end(text: str):
    if text.endswith("."):
        return text
    return text + "."


def clean(text: str):
    text = text.strip()
    text = append_punctuation_at_end(text)
    return text


def split_to_sentences(text: str) -> List[str]:
    sents = text.split(".")
    last = sents[-1]
    if last == "":
        sents.pop()
    return sents


def filter_less_than_word_count(count: int = 3):
    def filter_sentence(text: str) -> str:
        words = text.split(' ')
        return len(words) > count

    return filter_sentence


def pre_process(text: str) -> str:
    text = clean(text)
    sentences = split_to_sentences(text)
    filtered_sentences = filter(filter_less_than_word_count(), sentences)
    trimmed_sentences = map(lambda x: x.strip(), filtered_sentences)
    processed = '. '.join(trimmed_sentences)
    processed = append_punctuation_at_end(processed)
    return processed
