from kimm._src.kimm_export import kimm_export
from kimm._src.models.base_model import BaseModel


@kimm_export(parent_path=["kimm.utils"])
def get_reparameterized_model(model: BaseModel):
    if not hasattr(model, "get_reparameterized_model"):
        raise ValueError(
            "There is no 'get_reparameterized_model' method in the model. "
            f"Received: model type={type(model)}"
        )

    config = model.get_config()
    if config["reparameterized"] is True:
        return model

    config["reparameterized"] = True
    config["weights"] = None
    reparameterized_model = type(model).from_config(config)
    for layer, rep_layer in zip(model.layers, reparameterized_model.layers):
        if hasattr(layer, "get_reparameterized_weights"):
            kernel, bias = layer.get_reparameterized_weights()
            rep_layer.reparameterized_conv2d.kernel.assign(kernel)
            rep_layer.reparameterized_conv2d.bias.assign(bias)
        else:
            for weight, target_weight in zip(layer.weights, rep_layer.weights):
                target_weight.assign(weight)
    return reparameterized_model
