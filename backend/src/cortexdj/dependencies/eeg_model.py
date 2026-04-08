from typing import Annotated

from fastapi import Depends, Request

from cortexdj.ml.model import EEGNetClassifier


def get_eeg_model(request: Request) -> EEGNetClassifier | None:
    return getattr(request.app.state, "eeg_model", None)


EEGModelDep = Annotated[EEGNetClassifier | None, Depends(get_eeg_model)]
