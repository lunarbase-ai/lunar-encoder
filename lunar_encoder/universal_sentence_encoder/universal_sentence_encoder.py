import tensorflow_hub as hub


class UniversalSentenceEncoder:
    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def load(self, model_name: str):
        self.model = hub.load(model_name)

    def encode(self, sentences):
        return self.model(sentences).numpy().tolist()
