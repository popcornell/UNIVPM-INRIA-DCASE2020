from SED.baseline_tools.models.CRNN import CRNN
from SED.baseline_tools.utilities.Scaler import ScalerPerAudio, Scaler
import torch

def to_cuda_if_available(*args):
    """ Transfer object (Module, Tensor) to GPU if GPU available
    Args:
        args: torch object to put on cuda if available (needs to have object.cuda() defined)

    Returns:
        Objects on GPU if GPUs available
    """
    res = list(args)
    if torch.cuda.is_available():
        for i, torch_obj in enumerate(args):
            res[i] = torch_obj.cuda()
    if len(res) == 1:
        return res[0]
    return res


def _load_crnn(state):
    crnn_args = state["model"]["args"]
    crnn_kwargs = state["model"]["kwargs"]
    crnn = CRNN(*crnn_args, **crnn_kwargs)
    crnn.load(parameters=state["model"]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    print("Model loaded at epoch: {}".format(state["epoch"]))
    print(crnn)
    return crnn


def _load_scaler(state):
    scaler_state = state["scaler"]
    type_sc = scaler_state["type"]
    if type_sc == "ScalerPerAudio":
        scaler = ScalerPerAudio(*scaler_state["args"])
    elif type_sc == "Scaler":
        scaler = Scaler()
    else:
        raise NotImplementedError("Not the right type of Scaler has been saved in state")
    scaler.load_state_dict(state["scaler"]["state_dict"])
    return scaler