import nltk
downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for item in downloads:
    nltk.download(item)