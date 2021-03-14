from translation_dataset import TranslationDataset
import transformers

en_data_path = 'corpus.en_ru.1m.en'
rus_data_path = 'corpus.en_ru.1m.ru'

rus_tokenizer = transformers.BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')
en_tokenizer = transformers.RobertaTokenizerFast.from_pretrained('roberta-base')

train_dataset = TranslationDataset(en_data_path, rus_data_path, en_tokenizer, rus_tokenizer, 512, 512, 0, 1000)
input_tokens, input_mask, starget_tokens, target_mask = train_dataset.read_files()

print(input_tokens.shape)

