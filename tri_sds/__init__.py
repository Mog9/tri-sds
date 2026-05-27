def create_plugin():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from tri_sds.worker import TriSDSWorker

    HookRegistry.register(
        "sglang.srt.speculative.standalone_worker.StandaloneWorker",
        TriSDSWorker,
        HookType.REPLACE,
    )
