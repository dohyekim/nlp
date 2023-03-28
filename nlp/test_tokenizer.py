from transformers import AutoTokenizer

string = "첫 회를 시작으로 <13일:PRE>까지 <4일간:LOG> 총 <4회:PLA>에 걸쳐 매 회 <2편:VIT>씩 총 <8편:INP>이 공개될 예정이다."

string2 = "이것은 아무말이다. 오늘도 바쁠 것 같다."

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
text_tokenized = tokenizer(
    string,
    padding='max_length',
    max_length=512,
    truncation=True,
    return_tensors='pt'
)

print("tokenized_text: ", text_tokenized.input_ids.shape)

print("decoded text: ", tokenizer.decode(text_tokenized.input_ids[0]))

tokenized2 = tokenizer(
    [string, string2],
    padding='max_length',
    max_length=512,
    truncation=True,
    return_tensors='pt'
)

print("tokenized2: ", tokenized2.input_ids.shape)
print("tokenized2 decoded text: ", tokenizer.decode(tokenized2.input_ids[0]))
print("tokenized2 decoded text2: ", tokenizer.decode(tokenized2.input_ids[1]))


# ids to token (not text)

print("tokenized2 tokens: ", tokenizer.convert_ids_to_tokens(tokenized2.input_ids[1])) # '[CLS]', '이것', '##은', '아무', '##말', '##이다', '.', '오늘', '##도', '바', '##쁠', '것', '같', '##다', '.', '[SEP]', '[PAD]', '[PAD]'

# word_ids (subword token도 하나의 토큰으로)
print("tokenized2 tokens: ", tokenizer.convert_ids_to_tokens(tokenized2.input_ids[0])) # ''[CLS]', '첫', '회', '##를', '시작', '##으로', '<', '13', '##일', ':', 'PR', '##E', '>', '까지', '<', '4', '##일', '##간', ':', 'LO', '##G', '>', '총', '<', '4', '##회', ':', 'PL', '##A', '>', '에', '걸쳐', '매', '회', '<', '2', '##편', ':', 'V', '##IT', '>', '씩', '총', '<', '8', '##편', ':', 'IN', '##P', '>', '이', '공개', '##될', '예정', '##이다', '.', '[SEP]', '[PAD]', '[PAD]
word_ids = tokenized2.word_ids(0) # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
print("word_ids: ", word_ids) # [None, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9, 10, 10, 10, 11, 12, 12, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 21, 22, 23, 24, 25, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 37, 37, 38, 38, 39, None

print("tokenized2 tokens: ", tokenizer.convert_ids_to_tokens(tokenized2.input_ids[1])) # '[CLS]', '이것', '##은', '아무', '##말', '##이다', '.', '오늘', '##도', '바', '##쁠', '것', '같', '##다', '.', '[SEP]', '[PAD]', '[PAD]'
word_ids = tokenized2.word_ids(1) # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
print("word_ids: ", word_ids) # [None, 0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7, None, None, None, None, None, None, None, None, None, None, None, None, Non...] # CLS 이것은 아무말이다  오늘도 바쁠 것 같다 .