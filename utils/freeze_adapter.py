from peft import PeftModel

def freeze_adapter(model: PeftModel, adapter_name: str):
    """
    Freezes the parameters of a specific adapter in a PEFT model.

    Args:
        model (PeftModel): The PEFT model containing the adapter.
        adapter_name (str): The name of the adapter to freeze.
    """
    for name, param in model.named_parameters():
        if adapter_name in name:  # Check if the parameter belongs to the adapter
            param.requires_grad = False


def unfreeze_adapter(model: PeftModel, adapter_name: str):
    """
    Un-Freezes the parameters of a specific adapter in a PEFT model.

    Args:
        model (PeftModel): The PEFT model containing the adapter.
        adapter_name (str): The name of the adapter to freeze.
    """
    for name, param in model.named_parameters():
        if adapter_name in name:  # Check if the parameter belongs to the adapter
            param.requires_grad = True