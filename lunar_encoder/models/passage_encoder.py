import logging
from abc import ABC
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from tqdm.autonotebook import trange

from lunar_encoder.config import EncoderConfig
from lunar_encoder.models.base_encoder import BaseEncoder
from lunar_encoder.modules.dense import Dense
from lunar_encoder.modules.pooling import Pooling
from lunar_encoder.modules.transformer import Transformer
from lunar_encoder.utils import dict_batch_to_device, setup_logger

logger = logging.getLogger()
setup_logger(logger)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PassageEncoder(BaseEncoder):
    def __init__(self, config: Optional[EncoderConfig] = None):
        super(PassageEncoder, self).__init__(config)

        self._transformer = Transformer(
            model_name_or_path=self._config.base_transformer_name,
            max_seq_length=self._config.max_seq_len,
            model_args=self._config.base_transformer_args,
            tokenizer_name_or_path=self._config.base_transformer_name,
            tokenizer_args=self._config.base_tokenizer_args,
            do_lower_case=self._config.base_tokenizer_lowercase,
            cache_dir=self._config.cache_folder,
        ).to(self._config.device)

        self._pooler = Pooling(
            word_embedding_dimension=self._transformer.get_word_embedding_dimension(),
            pooling_method=self._config.pooling_method,
            pooled_embedding_name=self._config.pooled_embedding_name,
            **self._config.pooled_attention_params
        ).to(self._config.device)

        self._dense = None
        if self._config.add_dense:
            self._dense = Dense(
                input_dim=self._transformer.get_word_embedding_dimension(),
                output_dim=self._config.dense_output_dim
                if self._config.dense_output_dim is not None
                else self._transformer.get_word_embedding_dimension(),
                hidden_dims=self._config.dense_hidden_dims,
                activation=self._config.dense_activation,
                pooled_embedding_name=self._config.pooled_embedding_name,
            ).to(self._config.device)

    @property
    def transformer(self):
        return self._transformer

    @property
    def pooler(self):
        return self._pooler

    @property
    def dense(self):
        return self._dense

    def get_word_embedding_dimension(self) -> int:
        return self._transformer.get_word_embedding_dimension()

    def forward(self, features: Dict[str, Tensor]):
        features.update(self._transformer(features))
        features.update(self._pooler(features))
        if self._dense is not None:
            features.update(self._dense(features))
        return features

    def encode(
        self,
        input_instances: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        as_tensor: bool = False,
        normalize_embeddings: bool = False,
        use16: bool = True,
    ) -> Union[np.ndarray, Tensor]:

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() == logging.DEBUG

        input_was_string = False
        if isinstance(input_instances, str) or not hasattr(input_instances, "__len__"):
            input_instances = [input_instances]
            input_was_string = True

        self.to(self._config.device)

        all_embeddings = []
        for start_index in trange(
            0,
            len(input_instances),
            batch_size,
            desc="Batches",
            disable=not show_progress_bar,
        ):
            sentences_batch = input_instances[start_index : start_index + batch_size]
            features = self._transformer.tokenize(sentences_batch)
            features = dict_batch_to_device(features, self._config.device)

            with autocast(enabled=use16):
                with torch.no_grad():
                    out_features = self.forward(features)
                    embeddings = out_features[
                        self._config.pooled_embedding_name
                    ].detach()

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            if as_tensor:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = torch.stack(all_embeddings)

        if not as_tensor:
            all_embeddings = all_embeddings.cpu().numpy()

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        return self._transformer.tokenize(texts)
