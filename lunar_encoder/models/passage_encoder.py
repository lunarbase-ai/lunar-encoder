import json
import logging
import os
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
from lunar_encoder.utils import (
    dict_batch_to_device,
    load_module,
    save_module,
    setup_logger,
)

logger = logging.getLogger()
setup_logger(logger)


class PassageEncoder(BaseEncoder):
    def __init__(
        self,
        config: Optional[EncoderConfig] = None,
        pooling: Optional[Pooling] = None,
        dense: Optional[Dense] = None,
        device: str = "cpu",
    ):
        super(PassageEncoder, self).__init__(config, device)

        self._transformer = Transformer(
            model_name_or_path=self._config.base_transformer_name,
            max_seq_length=self._config.max_seq_length,
            model_args=self._config.base_transformer_args,
            tokenizer_args=self._config.base_tokenizer_args,
            cache_dir=self._config.cache_folder,
        )

        self._pooler = pooling
        if self._pooler is None:
            self._pooler = Pooling(
                pooling_method=self._config.pooling_method,
                pooled_embedding_name=self._config.pooled_embedding_name,
                **self._config.pooled_attention_params
            )

        self._dense = dense
        if (
            self._dense is None
            and self._config.dense_hidden_dims is not None
            and self._config.dense_hidden_dims > 0
        ):
            self._dense = Dense(
                input_dim=self._transformer.get_word_embedding_dimension(),
                output_dim=self._config.dense_output_dim
                if self._config.dense_output_dim is not None
                else self._transformer.get_word_embedding_dimension(),
                hidden_dims=self._config.dense_hidden_dims,
                activation=self._config.dense_activation,
                pooled_embedding_name=self._config.pooled_embedding_name,
            )

        self.to(self.device)

    @property
    def transformer(self):
        return self._transformer

    @property
    def pooler(self):
        return self._pooler

    @property
    def dense(self):
        return self._dense

    @transformer.setter
    def transformer(self, value: Transformer):
        self._transformer = value

    @pooler.setter
    def pooler(self, value: Pooling):
        self._pooler = value

    @dense.setter
    def dense(self, value: Dense):
        self._dense = value

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
        show_progress_bar: bool = False,
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

    def save(self, model_path: str):

        if model_path is None:
            return

        os.makedirs(model_path, exist_ok=True)

        logger.info("Saving model to {}".format(model_path))
        self._config.update(
            {"base_transformer_name": os.path.join(model_path, "transformer")}
        )
        with open(os.path.join(model_path, "passage_encoder_config.json"), "w") as fOut:
            json.dump(self._config, fOut, indent=2)

        self._transformer.save(os.path.join(model_path, "transformer"))
        save_module(
            self._pooler,
            os.path.join(model_path, "pooler", "passage_encoder_pooler.pt"),
        )
        if self._dense is not None:
            save_module(
                self._dense,
                os.path.join(model_path, "dense", "passage_encoder_dense.pt"),
            )

    @staticmethod
    def load(model_path: str):
        if not os.path.isdir(model_path):
            raise FileNotFoundError("{}: no such directory!".format(model_path))

        logger.info("Loading model from {}".format(model_path))
        with open(os.path.join(model_path, "passage_encoder_config.json")) as fIn:
            config = json.load(fIn)

        pooler = load_module(
            os.path.join(model_path, "pooler", "passage_encoder_pooler.pt")
        )
        dense = None
        if os.path.isfile(
            os.path.join(model_path, "dense", "passage_encoder_dense.pt")
        ):
            dense = load_module(
                os.path.join(model_path, "dense", "passage_encoder_dense.pt")
            )
        return PassageEncoder(config=config, pooling=pooler, dense=dense)
