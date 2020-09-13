import spacy
import pytextrank

nlp = spacy.load("en_core_web_sm")
text_rank = pytextrank.TextRank()
nlp.add_pipe(text_rank.PipelineComponent, name="textrank", last=True)

def keywords_review(review):
    doc = nlp(review)

    for p in doc._.phrases[:2]:
        print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
        print(p.chunks)
