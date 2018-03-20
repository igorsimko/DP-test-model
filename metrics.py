class Metrics(object):

    def __init__(self) -> None:
        self.ROUGE = []
        self.BLEU = []
        self.SIMILARITY = []

    def reset(self):
        self.ROUGE = []
        self.BLEU = []
        self.SIMILARITY = []