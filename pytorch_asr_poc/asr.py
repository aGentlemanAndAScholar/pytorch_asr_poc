import torch
import torchaudio
from importlib import import_module
from functools import lru_cache


class GreedyCTCDecoder(torch.nn.Module):
    """Decoder class for converting symbols to transcripts

    Source: `https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html#generating-transcripts`
    """

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


@lru_cache(maxsize=1)
def _get_bundle_model(model_name: str, device: str) -> (str, str):
    """
    Fetches and prepares pytorch model for ASR.

    Args:
        model_name: Name of pretrained pytorch wave2vec2 model - any value
        from `https://pytorch.org/audio/stable/pipelines.html#id36` should work
        device: Device running pytorch model

    Returns: Tuple of pytorch bundle and downloaded model object.
    """
    bundle = getattr(import_module("torchaudio.pipelines"), model_name)
    model = bundle.get_model().to(device)
    return bundle, model


def estimate_transcript(
    filename: str, model_name="WAV2VEC2_ASR_BASE_960H", device="cpu"
) -> str:
    """Stripped down implementation of ASR pipeline.

    Source: https://pytorch.org/audio/stable/pipelines.html#id36

    Args:
        filename: Full path of the wav file.
        model_name: Name of the pretrained model
        device: Defines type of device running the model

    Returns: Estimated transcript

    """
    bundle, model = _get_bundle_model(model_name=model_name, device=device)

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    waveform, sample_rate = torchaudio.load(filename)

    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, bundle.sample_rate
        )

    with torch.inference_mode():
        emission, _ = model(waveform)

    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])

    return transcript
