from typing import Annotated

from fastapi import Depends, Request

from cortexdj.ml.predict import EEGModel


def get_eeg_model(request: Request) -> EEGModel | None:
    return getattr(request.app.state, "eeg_model", None)


EEGModelDep = Annotated[EEGModel | None, Depends(get_eeg_model)]
