import transformers
import tensorflow as tf


class TranslationDataset:
    def __init__(self, input_file, target_file, input_tokenizer, target_tokenizer,
                 input_max_length, target_max_length, start=None, finish=None):

        self.input_file = input_file
        self.target_file = target_file
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length

        if start is None:
            self.start = 0
        else:
            self.start = start

        self.finish = finish

        self.input_tokens = None
        self.input_mask = None
        self.target_tokens = None
        self.target_mask = None

    def read_files(self):

        input_sentences = []
        target_sentences = []

        # Read input file
        with open(self.input_file, 'r') as f:
            for index, line in enumerate(f.readlines()):
                if self.finish is None:
                    if index >= self.start:
                        input_sentences.append(line)
                    else:
                        pass

                else:
                    if self.start <= index < self.finish:
                        input_sentences.append(line)

        # Read target file
        with open(self.target_file, 'r') as f:
            for index, line in enumerate(f.readlines()):
                if self.finish is None:
                    if index >= self.start:
                        target_sentences.append(line)
                    else:
                        pass

                else:
                    if self.start <= index < self.finish:
                        target_sentences.append(line)

        token_mask_input = self.input_tokenizer(input_sentences, return_tensors='np', add_special_tokens=True,
                                                max_length=self.input_max_length, padding='longest')

        token_mask_target = self.target_tokenizer(target_sentences, return_tensors='np', add_special_tokens=True,
                                                max_length=self.input_max_length, padding='longest')

        self.input_tokens = token_mask_input['input_ids']
        self.input_mask = token_mask_input['attention_mask']

        self.target_tokens = token_mask_target['input_ids']
        self.target_mask = token_mask_target['attention_mask']

        assert (self.input_tokens.shape[0] == self.target_tokens.shape[0]), "Mismatch in input and target files' lengths"
        print(f"Read {self.input_tokens.shape[0]} lines from input and target files")

        return self.input_tokens, self.input_mask, self.target_tokens, self.target_mask

    # def __len__(self):
    #     return self.input_tokens.shape[0]
    #
    # def __getitem__(self, item):
    #     return self.input_tokens[item, :], self.input_mask[item, :], self.target_tokens[item, :], self.target_mask[item, :],














