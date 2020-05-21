import tensorflow_datasets as tfds

from helpers import Word_Index, one_hot, load

tokenizer = tfds.features.text.Tokenizer()
word_index = Word_Index()
model = load()

review = "best movie best actors high praise"
tokenized_review = tokenizer.tokenize(review)
encoded_review = [word_index.get_index(word) for word in tokenized_review]
one_hot_review = one_hot([encoded_review])

prediction = model.predict(one_hot_review)
print(prediction)
