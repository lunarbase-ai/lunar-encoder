import dataclasses
import json
import logging
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from lunar_encoder.models.base_encoder import BaseEncoder
from lunar_encoder.models.config import ConfigSerializationEncoder, EncoderConfig
from lunar_encoder.models.lunar_typing.passage_trainer import PassageTrainer
from lunar_encoder.models.modules.dense import Dense
from lunar_encoder.models.modules.pooling import Pooling
from lunar_encoder.models.modules.transformer import Transformer
from lunar_encoder.utils import (
    dict_batch_to_device,
    get_parameter_names,
    load_module,
    save_module,
    unpack_batch,
)

NATIVE_CPU_AMP = False
if version.parse(torch.__version__) >= version.parse("1.10"):
    NATIVE_CPU_AMP = True


class PassageEncoder(BaseEncoder):
    def __init__(
        self,
        config: Optional[Union[str, EncoderConfig]] = None,
        pooling: Optional[Pooling] = None,
        dense: Optional[Dense] = None,
    ):
        super(PassageEncoder, self).__init__(config)

        self._trainer = None  # Until fit runs

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
                **self._config.pooled_attention_params,
            )

        self._dense = dense
        if self._dense is None and self._config.dense_hidden_dims is not None:
            self._dense = Dense(
                input_dim=self._transformer.get_word_embedding_dimension(),
                output_dim=self._config.dense_output_dim
                if self._config.dense_output_dim is not None
                else self._transformer.get_word_embedding_dimension(),
                hidden_dims=self._config.dense_hidden_dims,
                activation=self._config.dense_activation,
            )
        self.to(self.config.device)

    @property
    def trainer(self):
        return self._trainer

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

    def get_output_dimension(self) -> int:
        return (
            self._dense.output_dim
            if self._dense is not None
            else self.get_word_embedding_dimension()
        )

    def forward(self, features: Dict[str, Tensor], min_out: bool = True):
        """
        Defines the forward pass.

        Parameters
        ----------
        features: Dict[str, Tensor]
            Much like the format of Huggingface's tokenizer output.
        min_out: bool
            If True only the pooled embedding will be returned.

        Returns
        -------

        """
        features.update(self._transformer(features))
        features.update({self.config.pooled_embedding_name: self._pooler(features)})
        if self._dense is not None:
            features.update(
                {
                    self.config.pooled_embedding_name: self._dense(
                        features[self.config.pooled_embedding_name]
                    )
                }
            )
        if min_out:
            return {
                self.config.pooled_embedding_name: features[
                    self.config.pooled_embedding_name
                ]
            }
        return features

    def encode(
        self,
        input_instances: Union[str, Iterable[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        as_tensor: bool = False,
        normalize_embeddings: bool = True,
        amp: Optional[bool] = None,
    ) -> Union[np.ndarray, Tensor]:

        if amp is None:
            amp = self.config.amp

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = self.logger.getEffectiveLevel() == logging.DEBUG

        input_was_string = False
        if isinstance(input_instances, str) or not hasattr(input_instances, "__len__"):
            input_instances = [input_instances]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in input_instances])
        sentences_sorted = [input_instances[idx] for idx in length_sorted_idx]
        for start_index in trange(
            0,
            len(sentences_sorted),
            batch_size,
            desc="Batches",
            disable=not show_progress_bar,
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self._transformer.tokenize(sentences_batch)
            features = dict_batch_to_device(features, self.config.device)

            if NATIVE_CPU_AMP:
                with torch.autocast(
                    device_type=str(self.config.device),
                    enabled=amp,
                    dtype=torch.bfloat16
                    if self.config.device == "cpu"
                    else torch.float16,
                ):
                    with torch.no_grad():
                        out_features = self.forward(features)
                        embeddings = out_features[
                            self._config.pooled_embedding_name
                        ].detach()
            else:
                if self.config.device == "cpu" and amp:
                    self.logger.warning(
                        "Tried to use `AMP` but it is not supported on cpu with current PyTorch version."
                    )
                    with torch.no_grad():
                        out_features = self.forward(features)
                        embeddings = out_features[
                            self._config.pooled_embedding_name
                        ].detach()
                else:
                    with torch.cuda.amp.autocast(
                        enabled=amp,
                    ):
                        with torch.no_grad():
                            out_features = self.forward(features)
                            embeddings = out_features[
                                self._config.pooled_embedding_name
                            ].detach()

            if normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            if not as_tensor:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = torch.stack(all_embeddings).half()

        if not as_tensor:
            all_embeddings = all_embeddings.cpu().numpy()

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        return self._transformer.tokenize(texts)

    def init_trainer(self, trainer_config: Optional[Dict] = None):
        if trainer_config is None:
            self._trainer = PassageTrainer()
        else:
            self._trainer = PassageTrainer(**trainer_config)

        # Instantiate optimizer
        decay_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self._trainer.optimizer_args.get("weight_decay", 0.0),
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        self._trainer.optimizer = self._trainer.optimizer(
            optimizer_grouped_parameters, **self._trainer.optimizer_args
        )

        # Instantiate scheduler

        # Instantiate scaler
        if (
            self._trainer.scaler is None
            and self._config.amp
            and self.config.device != "cpu"
        ):
            self._trainer.scaler = GradScaler()

    def collate_tokenize(self, text_batch: List[Tuple[List[str], Optional[int]]]):
        """
        <([anchor, positive|negative], label)> or
        <([anchor, positive], )> or
        <([anchor, positive, negative], )> - these will repeat the anchor.
            A more efficient option would be to create a separate collation function for triplets.
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

        tokenized = self._transformer.tokenize(anchors + examples)
        tokenized = dict_batch_to_device(tokenized, self.config.device)

        return tokenized, num_positives

    def training_step(self, batch_data: Dict[str, torch.Tensor], num_positives: int):
        if NATIVE_CPU_AMP:
            with torch.autocast(
                device_type=str(self.config.device),
                enabled=self._config.amp,
                dtype=torch.bfloat16 if self.config.device == "cpu" else torch.float16,
            ):
                with torch.set_grad_enabled(True):
                    packed_features = self(batch_data)
                    unpacked_features = unpack_batch(packed_features, 2)
                    anchors, examples = unpacked_features[
                        self.config.pooled_embedding_name
                    ]

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
                    with torch.set_grad_enabled(True):
                        packed_features = self(batch_data)
                        unpacked_features = unpack_batch(packed_features, 2)
                        anchors, examples = unpacked_features[
                            self.config.pooled_embedding_name
                        ]

                        loss_values = self._trainer.loss(
                            anchors=anchors,
                            examples=examples,
                            num_positives=num_positives,
                        )

        if self._trainer.grad_accumulation > 1:
            loss_values = loss_values / self._trainer.grad_accumulation
        if self._trainer.scaler is None:
            loss_values.backward()
        else:
            self._trainer.scaler.scale(loss_values).backward()
        return loss_values.detach()

    def training_epoch(
        self,
        epoch_iterator: DataLoader,
        epoch_id: int,
        num_steps: Optional[int] = None,
        eval_iterator: Optional[DataLoader] = None,
        eval_callback: Optional[callable] = None,
    ):
        self.train()

        epoch_losses = []
        reference_eval_metric = None
        training_step_id = 0
        for batch_data, num_positives in tqdm(
            epoch_iterator,
            desc=f"\tIteration {epoch_id} ",
            smoothing=0.05,
            # disable=self.logger.getEffectiveLevel() != logging.DEBUG,
            disable=False,
        ):
            loss_values = self.training_step(
                batch_data=batch_data,
                num_positives=num_positives,
            )

            # if loss is nan or inf simply add the average of previous logged losses
            if torch.isnan(loss_values) or torch.isinf(loss_values):
                if len(epoch_losses) == 0:
                    raise ValueError(
                        f"Encountered {loss_values.item()} loss value and there are "
                        f"no accumulated losses for the current epoch."
                    )

                self.logger.warning(
                    f"Encountered {loss_values.item()} loss value at step {training_step_id}."
                )
                epoch_losses.append(np.mean(epoch_losses))
            else:
                epoch_losses.append(loss_values.item())

            if (
                self._trainer.grad_accumulation <= 1
                or (training_step_id + 1) % self._trainer.grad_accumulation == 0
                or (
                    self._trainer.grad_accumulation
                    >= num_steps
                    == (training_step_id + 1)
                )
            ):
                if self._trainer.scaler is not None:
                    self._trainer.scaler.unscale_(self._trainer.optimizer)
                if self._trainer.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self._trainer.max_grad_norm
                    )

                optimizer_was_run = True
                if self._trainer.scaler is not None:
                    scale_before = self._trainer.scaler.get_scale()
                    self._trainer.scaler.step(self._trainer.optimizer)
                    self._trainer.scaler.update()
                    optimizer_was_run = scale_before <= self._trainer.scaler.get_scale()
                else:
                    self._trainer.optimizer.step()

                if optimizer_was_run and self._trainer.scheduler is not None:
                    self._trainer.scheduler.step()

                self.zero_grad()

            training_step_id += 1

            # Checkpoint
            if training_step_id % self._trainer.checkpoint_steps == 0:
                if all(
                    [
                        param is not None
                        for param in [
                            eval_callback,
                            self._trainer.checkpoint_path,
                            eval_iterator,
                        ]
                    ]
                ):
                    self.eval()
                    is_better = eval_callback(
                        self, eval_iterator, reference_eval_metric
                    )
                    self.train()
                    if is_better:
                        self._save_checkpoint(
                            training_step_id, self._trainer.checkpoint_path
                        )
                elif self._trainer.checkpoint_path is not None:
                    self._save_checkpoint(
                        training_step_id, self._trainer.checkpoint_path
                    )

        return epoch_losses

    def fit(
        self,
        training_dataset: List[Tuple[List[str], Optional[int]]],
        fit_config: Optional[Dict] = None,
    ):
        """
        The training data is expected to come in as a collection of pairs or triplets.
        """

        if not all(
            [
                isinstance(training_instance, tuple)
                for training_instance in training_dataset
            ]
        ):
            raise ValueError(
                "Each training instance should be a tuple of the form <([anchor, positive|negative], label)> or "
                "<([anchor, positive], )> or "
                "<([anchor, positive, negative], )>"
            )

        self.init_trainer(fit_config)

        dataloader = DataLoader(
            training_dataset,
            batch_size=self._trainer.batch_size,
            shuffle=True,
            collate_fn=self.collate_tokenize,
        )
        num_steps = math.ceil(len(training_dataset) / self._trainer.batch_size)
        loss_log = dict()
        for e in range(self._trainer.epochs):
            epoch_losses = self.training_epoch(
                dataloader, epoch_id=e + 1, num_steps=num_steps
            )
            loss_log[e] = np.array(epoch_losses).mean()

        if self._trainer.checkpoint_path is None:
            self.save(os.path.join(self.config.cache_folder, '0'))

        return loss_log

    def save(self, model_path: str):
        if model_path is None:
            return

        model_path = os.path.abspath(model_path)
        os.makedirs(model_path, exist_ok=True)

        self.logger.info("Saving model to {}".format(model_path))
        self._config.base_transformer_name = model_path

        with open(os.path.join(model_path, "encoder_config.json"), "w") as fOut:
            json.dump(
                dataclasses.asdict(self._config),
                fOut,
                indent=2,
                cls=ConfigSerializationEncoder,
            )

        self._transformer.save(model_path)
        save_module(
            self._pooler,
            os.path.join(model_path, "pooler.pt"),
        )
        if self._dense is not None:
            save_module(
                self._dense,
                os.path.join(model_path, "dense.pt"),
            )

    @staticmethod
    def load(model_path: str):
        model_path = os.path.abspath(model_path)
        if not os.path.isdir(model_path):
            raise FileNotFoundError("{}: no such directory!".format(model_path))

        pooler = load_module(os.path.join(model_path, "pooler.pt"))
        dense = None
        if os.path.isfile(os.path.join(model_path, "dense.pt")):
            dense = load_module(os.path.join(model_path, "dense.pt"))

        config = PassageEncoder.load_json_config(
            os.path.join(model_path, "encoder_config.json")
        )
        config.cache_folder = model_path
        return PassageEncoder(
            config=config,
            pooling=pooler,
            dense=dense,
        )
