from pydantic import Field
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import logging

from lunar_encoder.utils import mean_pooling

LOGGER = logging.getLogger(__name__)


# Class used to encode sentences using HuggingFace's transformers library
class GenericTransformerEncoder:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


    def load(self, tokenizer_name: str, model_name: str):
        if tokenizer_name is not None:
            self.load_tokenizer(tokenizer_name=tokenizer_name)
        elif model_name is not None:
            self.load_tokenizer(tokenizer_name=model_name)
        if model_name is not None:
            self.load_model(model_name=model_name)


    def load_model(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name)

    def load_tokenizer(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


    def encode(self, sentences: List[str]):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings.tolist()

# if __name__ == "__main__":
#
# async def main():
#     mlrun.set_env_from_file("../mlrun-nonprod.env")
#     project = mlrun.get_or_create_project("encoder", context="./", user_project=True)
#     serving_function = mlrun.code_to_function(
#         filename="./serving.py",
#         name="hugging-face-serving",
#         kind="serving",
#         image="mlrun/mlrun"
#     )
#     graph = serving_function.set_topology("flow", engine="async")
#     graph \
#         .to(handler="preprocess", name="preprocess") \
#         .to("mlrun.frameworks.huggingface.HuggingFaceModelServer",
#             name="lunar-encoder",
#             task="question-answering",
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_class="BertForQuestionAnswering",
#             tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
#             tokenizer_class="AutoTokenizer"
#             ) \
#         .to(handler="postprocess", name="postprocess")
#
#     # graph.plot(filename='graph', format='png')
#
#     project.set_function(serving_function)
#     project.save()
#     await asyncio.sleep(3)
#     server = serving_function.to_mock_server()
#
#     # Testing
#
#     response = server.test(path='/predict', body={
#         'question': 'Why are flamingos pink?',
#         'context': 'The natural color of flamingos is grey. Flamingos turn pink due to the fact that they eat shrimps.'
#     })
#
#     print('>>>', response)
#
# asyncio.run(main())
