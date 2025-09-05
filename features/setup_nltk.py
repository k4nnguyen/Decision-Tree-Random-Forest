import nltk
# Download các thư viện con cần thiết trong nltk
downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
for item in downloads:
    nltk.download(item)