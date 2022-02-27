# MLServer

[MLServer](https://mlserver.readthedocs.io/en/latest/) aims to provide an easy way to start serving your machine learning models through a REST and gRPC interface, fully compliant with [KFServing's V2 Dataplane spec](https://kserve.github.io/website/modelserving/inference_api/). The list of cool features include

* [Adaptive batching](https://mlserver.readthedocs.io/en/latest/user-guide/adaptive-batching.html), to group inference requests together on the fly.
* [Parallel Inference Serving](https://mlserver.readthedocs.io/en/latest/user-guide/parallel-inference.html), for vertical scaling across multiple models through a pool of inference workers.
* Multi-model serving to run multiple models within the same process
* Support for the standard [V2 Inference Protocol](https://kserve.github.io/website/modelserving/inference_api/) on both the gRPC and REST flavours
* Scalability with deployment in Kubernetes native frameworks, including [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/graph/protocols.html#v2-kfserving-protocol) and [KServe](https://kserve.github.io/website/modelserving/v1beta1/sklearn/v2/), where MLServer is the core Python inference server used to serve machine learning models.

[Inference runtimes](https://github.com/SeldonIO/MLServer/blob/master/docs/runtimes/index.md) allow you to define how your model should be used within MLServer. You can think of them as the backend glue between MLServer and your machine learning framework of choice. It also provides supports inference runtimes for many frameworks such as:

1. [Scikit-Learn](https://github.com/SeldonIO/MLServer/blob/master/runtimes/sklearn)
2. [XGBoost](https://github.com/SeldonIO/MLServer/blob/master/runtimes/xgboost)
3. [Spark MLib](https://github.com/SeldonIO/MLServer/blob/master/runtimes/mllib)
4. [LightGBM](https://github.com/SeldonIO/MLServer/blob/master/runtimes/lightgbm)
5. [Tempo](https://github.com/SeldonIO/tempo)
6. [MLflow](https://github.com/SeldonIO/MLServer/blob/master/runtimes/mlflow)
7. [Writing custom runtimes](https://github.com/SeldonIO/MLServer/blob/master/docs/runtimes/custom.md)

In this exercise, we will deploy the sentiment analysis huggingface transformer model. Since MLServer does not provide out-of-the-box support for PyTorch or Transformer models, we will write a custom inference runtime to deploy this model.

```bash
pip install mlserver
# to install out-of-box frameworks
pip install mlserver-sklearn # or any of the frameworks supported above
```

## Custom Inference Runtime

It's very easy to extend MLServer for any framework other than the supported ones by writing a custom inference runtime. To add support for our framework, we extend `mlserver.MLModel` abstract class and overload two main methods:

* `load(self) -> bool`: Responsible for loading any artifacts related to a model (e.g. model weights, pickle files, etc.).
* `predict(self, payload: InferenceRequest) -> InferenceResponse`: Responsible for using a model to perform inference on an incoming data point.

```python
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
```

### Settings files

The next step will be to create 2 configuration files:

* `settings.json`: holds the configuration of our server (e.g. ports, log level, etc.).
* `model-settings.json`: holds the configuration of our model (e.g. input type, runtime to use, etc.).

## Run

### Locally

Test the sentiment classifier model

```bash
docker build -t sentiment -f sentiment/Dockerfile.sentiment sentiment/
docker run --rm -it sentiment
```

Test MLServer locally

```bash
# download trained models
bash get_models.sh
# create a docker image
mlserver build . -t 'sentiment-app:1.0.0'
docker run -it --rm -p 8080:8080 -p 8081:8081 sentiment-app:1.0.0
```

In a separate terminal,

```bash
# test inference request (REST)
python3  test_local_http_endpoint.py
# test inference request (gRPC)
python3  test_local_http_endpoint.py
```

### Additional Exercise

* Deploy the MLServer application on [SeldonCore](https://docs.seldon.io/projects/seldon-core/en/latest/) or [KServe](https://kserve.github.io/website/modelserving/v1beta1/sklearn/v2/).
