from collections import defaultdict
import json
from mlserver import types
from mlserver.model import MLModel
from mlserver.utils import get_model_uri
from mlserver.codecs import StringCodec

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class SentimentModel(MLModel):
    """
    Implementationof the MLModel interface to load and serve custom hugging face transformer models.
    """

    # load the model
    async def load(self) -> bool:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_uri = await get_model_uri(self._settings)

        self.model_name = model_uri
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        self.ready = True
        return self.ready

    # output predictions
    async def predict(self, payload: types.InferenceRequest) -> types.InferenceResponse:
        input_id, attention_mask = self._preprocess_inputs(payload)
        prediction = self._model_predict(input_id, attention_mask)

        return types.InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="predictions",
                    shape=prediction.shape,
                    datatype="FP32",
                    data=np.asarray(prediction).tolist(),
                )
            ],
        )

    # preprocess input payload
    def _preprocess_inputs(self, payload: types.InferenceRequest):
        inp_text = defaultdict()
        for inp in payload.inputs:
            inp_text[inp.name] = json.loads(
                "".join(self.decode(inp, default_codec=StringCodec))
            )
        inputs = self.tokenizer(inp_text['text'], return_tensors="pt")
        input_id = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        return input_id, attention_mask

    # run inference
    def _model_predict(self, input_id, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_id, attention_mask)
            probs = F.softmax(outputs.logits, dim=1).numpy()[0]
        return probs