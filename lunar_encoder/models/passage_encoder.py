import dataclasses
import json
import logging
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
from tqdm.notebook import tqdm

from lunar_encoder.models.base_encoder import BaseEncoder
from lunar_encoder.models.config import EncoderConfig
from lunar_encoder.models.modules.dense import Dense
from lunar_encoder.models.modules.pooling import Pooling
from lunar_encoder.models.modules.transformer import Transformer
from lunar_encoder.typing import Loss, Optimizer, Scheduler, PassageTrainer
from lunar_encoder.utils import (
    dict_batch_to_device,
    load_module,
    save_module,
    pack_batch,
    unpack_batch,
    get_parameter_names,
)


class PassageEncoder(BaseEncoder):
    def __init__(
        self,
        config: Optional[EncoderConfig] = None,
        pooling: Optional[Pooling] = None,
        dense: Optional[Dense] = None,
    ):
        super(PassageEncoder, self).__init__(config)

        self._trainer = PassageTrainer()

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

    def forward(self, features: Dict[str, Tensor], min_out: bool = True):
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
        use16: Optional[bool] = None,
    ) -> Union[np.ndarray, Tensor]:

        if use16 is None:
            use16 = self.config.use16

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
            with torch.autocast(
                device_type=self.config.device,
                enabled=use16,
                dtype=torch.bfloat16 if self.config.device == "cpu" else torch.float16,
            ):
                with torch.no_grad():
                    out_features = self.forward(features)
                    embeddings = out_features[
                        self._config.pooled_embedding_name
                    ].detach()

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            if not as_tensor:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = torch.stack(all_embeddings)

        if not as_tensor:
            all_embeddings = all_embeddings.cpu().numpy()

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        return self._transformer.tokenize(texts)

    def init_trainer(self):
        # Instantiate loss
        if isinstance(self._config.loss, str):
            self._config.loss = Loss[self._config.loss.upper()]
        self._config.loss_args.update({"distance_metric": self._config.distance_metric})
        self._trainer.loss = self._config.loss(**self._config.loss_args)

        # Instantiate optimizer
        if isinstance(self._config.optimizer, str):
            self._config.optimizer = Optimizer[self._config.optimizer.upper()]
        decay_parameters = get_parameter_names(self, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self._config.default_weight_decay,
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        self._trainer.optimizer = self._config.optimizer(
            optimizer_grouped_parameters, **self._config.optimizer_args
        )

        # Instantiate scheduler
        if isinstance(self._config.scheduler, str):
            self._config.scheduler = Scheduler[self._config.scheduler.upper()]
        self._trainer.scheduler = self._config.scheduler(
            self._trainer.optimizer, **self._config.scheduler_args
        )

        # Instantiate scaler
        self._trainer.scaler = None
        if self._config.amp and self.device != "cpu":
            self._trainer.scaler = GradScaler()

    def training_step(self, batch_data: List[Dict[str, torch.Tensor]]):
        with torch.autocast(
            device_type=self.device,
            enabled=self._config.amp,
            dtype=torch.bfloat16 if self.device == "cpu" else torch.float16,
        ):
            packed_batch = pack_batch(batch_data)
            with torch.set_grad_enabled(True):
                packed_features = self(packed_batch)
                unpacked_features = unpack_batch(packed_features, len(batch_data))
                loss_values = self._trainer.loss(
                    *unpacked_features[self.config.pooled_embedding_name]
                )

        if self._config.grad_accumulation > 1:
            loss_values = loss_values / self._config.grad_accumulation
        if self._trainer.scaler is None:
            loss_values.backward()
        else:
            self._trainer.scaler.scale(loss_values).backward()
        return loss_values.detach()

    def training_epoch(
        self, epoch_iterator: Iterator[List[Dict[str, torch.Tensor]]], num_steps: int
    ):
        self.train()

        epoch_losses = []
        training_step_id = 0
        for batch_data in tqdm(
            epoch_iterator,
            desc=f"\tIteration {training_step_id} ",
            smoothing=0.05,
            disable=self.logger.getEffectiveLevel() != logging.DEBUG,
        ):
            loss_values = self.training_step(batch_data=batch_data)

            # if loss is nan or inf simply add the average of previous logged losses
            if torch.isnan(loss_values) or torch.isinf(loss_values):
                if len(epoch_losses) == 0:
                    raise ValueError(
                        f"Encountered {loss_values.item()} loss value and there are no accumulated losses for the current epoch."
                    )

                self.logger.warning(
                    f"Encountered {loss_values.item()} loss value at step {training_step_id}."
                )
                epoch_losses.append(np.mean(epoch_losses))
            else:
                epoch_losses.append(loss_values.item())

            if (
                self._config.grad_accumulation <= 1
                or (training_step_id + 1) % self._config.grad_accumulation == 0
                or (
                    self._config.grad_accumulation
                    >= num_steps
                    == (training_step_id + 1)
                )
            ):
                if self._trainer.scaler is not None:
                    self._trainer.scaler.unscale_(self._trainer.optimizer)
                if self._config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self._config.max_grad_norm
                    )

                optimizer_was_run = True
                if self._trainer.scaler is not None:
                    scale_before = self._trainer.scaler.get_scale()
                    self._trainer.scaler.step(self._trainer.optimizer)
                    self._trainer.scaler.update()
                    optimizer_was_run = scale_before <= self._trainer.scaler.get_scale()
                else:
                    self._trainer.optimizer.step()

                if optimizer_was_run:
                    self._trainer.scheduler.step()

                self.zero_grad()

            training_step_id += 1

        return epoch_losses

    def fit(self, data_loader: DataLoader, batch_size: int = 32, num_epochs: int = 1):
        self.init_trainer()
        # TODO: Add main outer training loop + data/batch preparation


    def save(self, model_path: str):

        if model_path is None:
            return

        os.makedirs(model_path, exist_ok=True)

        self.logger.info("Saving model to {}".format(model_path))
        self._config.base_transformer_name = model_path

        with open(os.path.join(model_path, "passage_encoder_config.json"), "w") as fOut:
            json.dump(dataclasses.asdict(self._config), fOut, indent=2)

        self._transformer.save(model_path)
        save_module(
            self._pooler,
            os.path.join(model_path, "passage_encoder_pooler.pt"),
        )
        if self._dense is not None:
            save_module(
                self._dense,
                os.path.join(model_path, "passage_encoder_dense.pt"),
            )

    def load(self, model_path: str):
        if not os.path.isdir(model_path):
            raise FileNotFoundError("{}: no such directory!".format(model_path))

        self.logger.info("Loading model from {}".format(model_path))
        with open(os.path.join(model_path, "passage_encoder_config.json")) as fIn:
            config_dict = json.load(fIn)
        config = EncoderConfig()
        config.__dict__.update(config_dict)

        pooler = load_module(os.path.join(model_path, "passage_encoder_pooler.pt"))
        dense = None
        if os.path.isfile(os.path.join(model_path, "passage_encoder_dense.pt")):
            dense = load_module(os.path.join(model_path, "passage_encoder_dense.pt"))

        return PassageEncoder(config=config, pooling=pooler, dense=dense)
