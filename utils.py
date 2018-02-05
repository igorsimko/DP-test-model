import datetime
import nltk

def prt(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))

def bleu(reference, hypothesis):
    return nltk.translate.bleu(reference, hypothesis)