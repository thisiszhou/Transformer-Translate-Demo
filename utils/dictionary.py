from typing import Dict


class Dictionary(object):
    def __init__(self, language_dict: Dict[str, int]):
        self.word2ids = language_dict
        self.ids2word = {v: k for k, v in self.word2ids.items()}

    @property
    def w2i(self):
        return self.word2ids

    @property
    def i2w(self):
        return self.ids2word

    def __len__(self):
        return len(self.word2ids)




