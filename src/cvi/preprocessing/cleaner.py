import spacy
import re

nlp = spacy.load("en_core_web_sm",disable=["parser","ner"])

def clean_text(text):
    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+|https\S+","", text)

    # remove special characters & numbers
    text = re.sub(r"[^a=z\s]","",text)

    # process with spaCy
    doc =nlp(text)

    # lemmatization + remove stopwords

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    ]

    return " ".join(tokens)

def preprocess_texts(texts):
    cleaned_texts = []

    texts = [
        re.sub(r"http\S+|www\S+|https\S+","",str(text).lower())
        for text in texts
    ]

    for doc in nlp.pipe(texts,batch_size=500):
        tokens = [
            # "running", "run", "ran" -> all these have same meaning now token.lemma convert it into "run" only 
            token.lemma_ 
            for token in doc
            if not token.is_stop and token.is_alpha and len(token) > 2 
        ]
        cleaned_texts.append(" ".join(tokens))

    return cleaned_texts    