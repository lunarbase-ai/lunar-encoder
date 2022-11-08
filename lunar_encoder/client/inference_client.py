import logging
from typing import List, Optional, Union
from urllib.parse import urlparse

import httpx
import numpy as np
from httpx import HTTPError

from lunar_encoder.utils import setup_logger

logger = logging.getLogger()
setup_logger(logger)


def raise_on_not200(response):
    if response.status_code != 200:
        raise HTTPError(
            f"There was an error processing the request. Response code: {response.status_code}!"
        )


class InferenceClient:
    """
    TODO: Update after lunar-engine's EncoderClient. Or remove altogether.
    """
    def __init__(
        self,
        scheme: str = "http",
        hostname: str = "localhost",
        port: int = 8080,
        endpoint: str = "predictions",
        model_name: Optional[str] = None,
        default_timeout: int = 1000,
    ):
        """
        Create a LunarEncoder client object that connects to a LunarEncoder server.
        """

        _base_url = f"{scheme}://{hostname}:{port}/{endpoint}"
        self._model_name = model_name
        try:
            r = urlparse(_base_url)
        except:
            raise ValueError(f"{_base_url} is not a valid connection string!")

        _kwargs = dict(
            timeout=default_timeout,
            base_url=_base_url,
            event_hooks={"response": [raise_on_not200]},
        )
        self._client = httpx.Client(**_kwargs)

    def predict(
        self, input_data: Union[str, List[str]], model_name: Optional[str] = None
    ):
        if model_name is None:
            model_name = self._model_name
        if model_name is None:
            raise ValueError(f"Missing mandatory model name!")

        result = self._client.post(
            url=model_name, json=self.prepare_json_input(input_data)
        )
        return self.prepare_numpy_output(result.json())

    @staticmethod
    def prepare_json_input(input_data):
        if isinstance(input_data, str):
            input_data = [input_data]
        return input_data

    @staticmethod
    def prepare_numpy_output(response_data):
        response_data = np.array(response_data)
        return response_data


if __name__ == '__main__':
    client = InferenceClient(
        hostname="127.0.0.1",
        port=8443,
        model_name="lunarenc"
    )
    embeddings = client.predict("This is a test sentence.")
    print(embeddings)
