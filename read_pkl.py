import pickle

# 查看 vocab
with open("outputs/TinyStories_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
print(vocab)

# 查看 merges
with open("outputs/TinyStories_merges.pkl", "rb") as f:
    merges = pickle.load(f)
print(merges)