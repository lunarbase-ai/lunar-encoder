import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import Tensor

from lunar_encoder.models.base_encoder import BaseEncoder
from lunar_encoder.models.config import EncoderConfig
from lunar_encoder.models.modules.dense import Dense
from lunar_encoder.models.modules.pooling import Pooling
from lunar_encoder.models.passage_encoder import PassageEncoder
from lunar_encoder.utils import (
    dict_batch_to_device,
    unpack_batch,
)

NATIVE_CPU_AMP = False
if version.parse(torch.__version__) >= version.parse("1.10"):
    NATIVE_CPU_AMP = True


class DualPassageEncoder(PassageEncoder):
    """
    A dual passage encoder architecture with a trainable query encoder and a frozen context encoder.
    """

    def __init__(
        self,
        query_config: Optional[Union[str, EncoderConfig]] = None,
        context_config: Optional[Union[str, EncoderConfig]] = None,
        query_pooling: Optional[Pooling] = None,
        context_pooling: Optional[Pooling] = None,
        query_dense: Optional[Dense] = None,
        context_dense: Optional[Dense] = None,
    ):
        super(DualPassageEncoder, self).__init__(
            query_config, query_pooling, query_dense
        )

        if context_config is None:
            context_config = EncoderConfig()
        elif isinstance(context_config, str):
            context_config = BaseEncoder.load_json_config(context_config)
        else:
            context_config = context_config

        if len(context_config.base_tokenizer_args):
            context_config.base_tokenizer_args = dict()
            raise Warning(
                "Found context tokenizer configuration, "
                "however this will be ignored because a dual encoder must use the same tokenizer for both "
                "query and context!"
            )

        self._context_encoder = PassageEncoder(
            config=context_config, pooling=context_pooling, dense=context_dense
        )
        self._context_encoder.transformer.tokenizer = self.transformer.tokenizer

        self.__encoder_sanity_check()

    @property
    def context_encoder(self):
        return self._context_encoder

    def __encoder_sanity_check(self):
        if self.get_output_dimension() != self._context_encoder.get_output_dimension():
            raise ValueError(
                "Query and context transformers must have the same output dimensionality."
            )

        if (self._config.amp != self._context_encoder.config.amp) or (
            self.device != self._context_encoder.device
        ):
            raise ValueError(
                "Query and context encoders must use the same device with the same Automatic Mixed Precision configuration!"
            )

    def query_encode(
        self,
        input_instances: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        as_tensor: bool = False,
        normalize_embeddings: bool = True,
        amp: Optional[bool] = None,
    ) -> Union[np.ndarray, Tensor]:
        return self.encode(
            input_instances=input_instances,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            as_tensor=as_tensor,
            normalize_embeddings=normalize_embeddings,
            amp=amp,
        )

    def context_encode(
        self,
        input_instances: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        as_tensor: bool = False,
        normalize_embeddings: bool = True,
        amp: Optional[bool] = None,
    ) -> Union[np.ndarray, Tensor]:
        return self._context_encoder.encode(
            input_instances=input_instances,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            as_tensor=as_tensor,
            normalize_embeddings=normalize_embeddings,
            amp=amp,
        )

    def collate_tokenize(self, text_batch: List[Tuple[List[str], Optional[int]]]):
        """
        <([anchor, positive|negative], label)> or
        <([anchor, positive], )> or
        <([anchor, positive, negative], )>
        """

        with_labels = len(text_batch[0]) == 2
        if not with_labels:
            batch_len = len(text_batch)
            for i in range(batch_len):
                if len(text_batch[i][0]) == 2:
                    text_batch[i] = (text_batch[i][0], 1)
                elif len(text_batch[i][0]) == 3:
                    a, p, n = text_batch[i][0]
                    text_batch[i] = ([a, p], 1)
                    text_batch.append(([a, n], 0))

        anchors, examples = [], []
        anchors += [example[0][0] for example in text_batch if example[1] == 1]
        examples += [example[0][1] for example in text_batch if example[1] == 1]

        assert len(anchors) == len(
            examples
        ), "Encountered different numbers of positive anchors and positive examples:".format(
            len(anchors), len(examples)
        )
        num_positives = len(examples)

        anchors += [example[0][0] for example in text_batch if example[1] == 0]
        examples += [example[0][1] for example in text_batch if example[1] == 0]

        assert len(anchors) == len(
            examples
        ), "Encountered different numbers of anchors and examples:".format(
            len(anchors), len(examples)
        )

        # tokenized_anchors = self.tokenize(anchors)
        # tokenized_examples = self._context_encoder.tokenize(examples)
        # tokenized = pack_batch([tokenized_anchors, tokenized_examples])

        tokenized = self.tokenize(anchors + examples)
        tokenized = dict_batch_to_device(tokenized, self.config.device)

        return tokenized, num_positives

    def training_step(self, batch_data: Dict[str, torch.Tensor], num_positives: int):
        if NATIVE_CPU_AMP:
            with torch.autocast(
                device_type=str(self.config.device),
                enabled=self._config.amp,
                dtype=torch.bfloat16 if self.config.device == "cpu" else torch.float16,
            ):
                unpacked_features = unpack_batch(batch_data, 2)
                tokenized_anchors = {
                    k: unpacked_features[k][0] for k in unpacked_features.keys()
                }
                tokenized_examples = {
                    k: unpacked_features[k][1] for k in unpacked_features.keys()
                }

                with torch.set_grad_enabled(False):
                    context_features = self._context_encoder(tokenized_examples)
                    examples = context_features[self.config.pooled_embedding_name]
                with torch.set_grad_enabled(True):
                    anchor_features = self(tokenized_anchors)
                    anchors = anchor_features[self.config.pooled_embedding_name]

                    loss_values = self._trainer.loss(
                        anchors=anchors,
                        examples=examples,
                        num_positives=num_positives,
                    )
        else:
            if self.config.device == "cpu" and self.config.amp:
                raise ValueError(
                    "Tried to use `fp16` but it is not supported on cpu with current PyTorch version."
                )
            else:
                with torch.cuda.amp.autocast(
                    enabled=self._config.amp,
                ):
                    unpacked_features = unpack_batch(batch_data, 2)
                    tokenized_anchors = {
                        k: unpacked_features[k][0] for k in unpacked_features.keys()
                    }
                    tokenized_examples = {
                        k: unpacked_features[k][1] for k in unpacked_features.keys()
                    }
                    with torch.set_grad_enabled(False):
                        context_features = self._context_encoder(tokenized_examples)
                        examples = context_features[self.config.pooled_embedding_name]
                    with torch.set_grad_enabled(True):
                        anchor_features = self(tokenized_anchors)
                        anchors = anchor_features[self.config.pooled_embedding_name]

                        loss_values = self._trainer.loss(
                            anchors=anchors,
                            examples=examples,
                            num_positives=num_positives,
                        )

        if self._config.grad_accumulation > 1:
            loss_values = loss_values / self._config.grad_accumulation
        if self._trainer.scaler is None:
            loss_values.backward()
        else:
            self._trainer.scaler.scale(loss_values).backward()
        return loss_values.detach()

    def save(self, model_path: str):
        if model_path is None:
            return
        model_path = os.path.abspath(model_path)
        os.makedirs(model_path, exist_ok=True)

        super().save(os.path.join(model_path, "query_encoder"))
        self._context_encoder.save(os.path.join(model_path, "context_encoder"))

    @staticmethod
    def load(model_path: str):
        model_path = os.path.abspath(model_path)
        if not os.path.isdir(model_path):
            raise FileNotFoundError("{}: no such directory!".format(model_path))

        query_encoder = PassageEncoder.load(os.path.join(model_path, "query_encoder"))
        context_encoder = PassageEncoder.load(
            os.path.join(model_path, "context_encoder")
        )

        return DualPassageEncoder(
            query_config=query_encoder.config,
            context_config=context_encoder.config,
            query_pooling=query_encoder.pooler,
            context_pooling=context_encoder.pooler,
            query_dense=query_encoder.dense,
            context_dense=context_encoder.dense,
        )
