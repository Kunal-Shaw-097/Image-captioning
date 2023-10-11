import torch
import json
import re


class Tokenizer:
    def __init__(self, vocab_path) -> None:
        with open(vocab_path, "r") as stream :
            self.vocab = json.load(stream)
        self.len = self.vocab["vocab_size"]
        self.word_to_index = self.vocab["word_to_index"]
        self.index_to_word = self.vocab["index_to_word"]
        self.pad_id = self.word_to_index["<PAD>"]
        self.start_id = self.word_to_index["<S>"]
        self.end_id = self.word_to_index["</S>"]
        
    def __len__(self):
        return self.len


    def combos(self, s, first=False):  # returns all possible combinations of sub words for a given word
        if not s:
            return
        length = len(s)
        if not first:
            yield [s]
        for i in range(1, length):
            for c in self.combos(s[: length - i]):
                yield c + [s[length - i :]]


    def get_indx(self, combos: list):
        indx_of_combo = -1  # represents index of the combination of split in the list of combinations
        for i, combo in enumerate(combos):
            cond = True  # flag
            for ele in combo:
                if ele not in self.word_to_index:
                    cond = False  # set to false if any sub word if not in the vocab
                    break
            if cond:  # if all subword of a word in vocab this triggers
                indx_of_combo = i
                break
        indx = []
        if indx_of_combo != -1:  # check if the indx changed
            for ele in combos[indx_of_combo]:
                indx.append(self.word_to_index[ele])

        return indx  # returns a list of indx of the subwords present in vocab


    def split_word(self, word: str):  # return indx of the sub words
        combos = []
        for c in self.combos(word, True):
            size = len(c)

            c[size - 1] = (
                c[size - 1] + "</w>"
            )  # adding the word end token to the last subword of all combinations

            combos.append(c)
        combos = sorted(
            combos, key=lambda x: len(x)
        )  # sorting the combinations based on len
        indx = self.get_indx(combos)
        return indx


    def create_pad(self, x: list):
        max = 0
        
        for line in x:
            length = len(line)
            if length > max:
                max = length

        for i, line in enumerate(x):
            while len(line) < max:
                x[i].append(self.word_to_index["<PAD>"])
        return x


    def encode(self, x: list):
        encoded_x = []

        if isinstance(x, list):  # batched encoding
            for line in x:
                encoded_line = [
                    self.word_to_index["<S>"]
                ]  # adding starting token index
                for word in filter(
                    lambda x: x not in [None, ""], re.split("\s+|(\W{1})", line)
                ):
                    if len(re.findall("\W", word)) == 0:
                        word = f"{word}</w>"
                    if word in self.word_to_index:
                        encoded_line.append(self.word_to_index[word])
                    else:
                        y = self.split_word(word.replace("</w>", ""))
                        for indx in y:
                            encoded_line.append(indx)
                encoded_line.append(
                    self.word_to_index["</S>"]
                )  # adding ending token index
                encoded_x.append(encoded_line)
            encoded_x = self.create_pad(encoded_x)

        else:  # single encoding
            encoded_x.append(self.word_to_index["<S>"])  # adding starting token index
            for word in filter(
                lambda x: x not in [None, ""], re.split("\s+|(\W{1})", x)
            ):
                if len(re.findall("\W", word)) == 0:
                    word = f"{word}</w>"
                if word in self.word_to_index:
                    encoded_x.append(self.word_to_index[word])
                else:
                    y = self.split_word(word.replace("</w>", ""))
                    for indx in y:
                        encoded_x.append(indx)
            encoded_x.append(self.word_to_index["</S>"])  # adding ending token index

        return torch.tensor(encoded_x)

    def decode_step(self, x):
        y = []
        for indx in x:
            y.append(self.index_to_word[str(int(indx))])
        pop_indx = []  # indexes to be removed after they are merged into other words
        for i, word in enumerate(y):
            if word in ["<S>", "</S>", "<PAD>", "</w>"]:
                pop_indx.append(i)
                continue
            elif len(
                re.findall("\W", word.replace("</w>", ""))
            ):  # case of special characters
                y[i - 1] = y[i - 1] + y[i] # adding the special characters to the previous word
                pop_indx.append(i)
                continue
            elif (
                not word.endswith("</w>") and i < len(y) - 1
            ):  # case of subwords which are not the last word of a sentence
                y[i + 1] =  y[i] + y[i + 1]  # combine then to the next subword to create  the full word
                pop_indx.append(i)
                continue
            y[i] = word.replace("</w>", "")  # removing the end of word token
        for i in reversed(pop_indx):  # popping off all the merged indexes
            y.pop(i)
        return " ".join(y)

    def decode(self, x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()

        if isinstance(x[0], list):  # batched decoding
            y = []
            for line in x:
                y.append(self.decode_step(line))
            return y
        else:  # single decoding
            return self.decode_step(x)
