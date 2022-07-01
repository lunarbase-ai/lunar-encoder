"""
Inspired by https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
"""

from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config
from typing import List, Dict, Optional, Union, Tuple


class Transformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings."""

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: Optional[int] = None,
        model_args: Optional[Dict] = None,
        tokenizer_args: Optional[Dict] = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
        cache_dir: Optional[str] = None,
    ):
        super(Transformer, self).__init__()

        model_args = model_args if model_args is not None else {}
        tokenizer_args = tokenizer_args if tokenizer_args is not None else {}

        self._do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(
            model_name_or_path, **model_args, cache_dir=cache_dir
        )
        self._load_model(model_name_or_path, config, cache_dir)

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path
            if tokenizer_name_or_path is not None
            else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args
        )

        if max_seq_length is None:
            if (
                hasattr(self._auto_model, "config")
                and hasattr(self._auto_model.config, "max_position_embeddings")
                and hasattr(self._tokenizer, "model_max_length")
            ):
                max_seq_length = min(
                    self._auto_model.config.max_position_embeddings,
                    self._tokenizer.model_max_length,
                )

        self._max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self._auto_model.config.tokenizer_class = self._tokenizer.__class__.__name__

    def _load_model(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir)
        else:
            self._auto_model = AutoModel.from_pretrained(
                model_name_or_path, config=config, cache_dir=cache_dir
            )

    def _load_t5_model(self, model_name_or_path, config, cache_dir):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel

        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self._auto_model = T5EncoderModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir
        )

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(
            self.get_config_dict(), self._auto_model.__class__.__name__
        )

    def forward(self, features: Dict[str, Tensor]):
        trans_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "token_type_ids" in features:
            trans_features["token_type_ids"] = features["token_type_ids"]

        output_states = self._auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update(
            {
                "token_embeddings": output_tokens,
                "attention_mask": features["attention_mask"],
            }
        )

        if self._auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if (
                len(output_states) < 3
            ):  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self._auto_model.config.hidden_size

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self._do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self._tokenizer(
                *to_tokenize,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self._max_seq_length
            )
        )
        return output

    @property
    def auto_model(self):
        return self._auto_model

    @property
    def tokenizer(self):
        return self._tokenizer
