import datetime
import nltk
import numpy as np

EOS_EL = 2

def prt(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))

def bleu(reference, hypothesis):
    return nltk.translate.bleu(reference, hypothesis)

def get_eos_pos(arr, return_val):
    if arr.any():
        if not np.where(arr == EOS_EL) == False:
            return np.where(arr == EOS_EL)[0][0]
        else:
            return return_val
