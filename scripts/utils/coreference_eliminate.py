import neuralcoref
import spacy
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp, greedyness=.5)


def eliminate_coreference(text):
    doc = nlp(text)
    return doc._.coref_resolved


if __name__ == "__main__":
   res = eliminate_coreference('''
    The bullsnake (Pituophis catenifer sayi) is a large non-venomous colubrid snake. It is currently considered a subspecies of the gopher snake (Pituophis catenifer).''')
   print(res)
