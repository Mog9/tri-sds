import os

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.standalone_worker import StandaloneWorker

from tri_sds.triton_kernels.draft_step import DraftModelState, draft_model_forward


class TriSDSWorker(StandaloneWorker):
    """triton-based speculative decoding worker.

    sds_enabled=False: pure pass-through to StandaloneWorker (no behavior change).
    sds_enabled=True:  replaces draft forward with custom triton kernels.
    """

    sds_enabled: bool = False
    _draft_state: DraftModelState = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sds_enabled = os.environ.get("SDS_ENABLED", "0") == "1"

    def _init_draft_state(self):
        """build DraftModelState from the loaded draft model."""
        draft = getattr(self, "draft_model", None)
        if draft is None:
            draft = self.model
        self._draft_state = DraftModelState(draft)

    #dispatch 

    def forward_batch_generation(self, batch: ScheduleBatch):
        if not self.sds_enabled:
            return super().forward_batch_generation(batch)
        return self._forward_batch_generation_sds(batch)

    def draft(self, batch: ScheduleBatch):
        if not self.sds_enabled:
            return super().draft(batch)
        return self._draft_sds(batch)

    def verify(self, batch: ScheduleBatch):
        if not self.sds_enabled:
            return super().verify(batch)
        return self._verify_sds(batch)

    #SDS implementations

    def _forward_batch_generation_sds(self, batch: ScheduleBatch):
        """full generation with custom triton kernels (non-spec path)."""
        raise NotImplementedError("sds non-spec generation not yet implemented")

    def _draft_sds(self, batch: ScheduleBatch):
        """run one draft step with triton kernels."""
        if self._draft_state is None:
            self._init_draft_state()

        # extract input data from batch
        input_ids = batch.input_ids
        positions = batch.positions
        B, T = input_ids.shape

        logits = draft_model_forward(self._draft_state, input_ids, positions)
        return logits

    def _verify_sds(self, batch: ScheduleBatch):
        """verify draft tokens with triton kernels."""
        raise NotImplementedError("sds verify not yet implemented")
