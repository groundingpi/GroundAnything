"""Start bundled SGLang with read-only metadata RPCs ready before generation."""
import os
import sys


def install_metadata_readiness(mixin):
    """The bundled version initializes its reply loop lazily on generation.

    /server_info also needs that loop, including with --skip-server-warmup.
    Initializing it here preserves a read-only preflight without dummy inference.
    """
    original = mixin.get_internal_state
    if getattr(original, '_grounding_metadata_ready', False):
        return

    async def get_internal_state(self):
        self.auto_create_handle_loop()
        return await original(self)

    get_internal_state._grounding_metadata_ready = True
    mixin.get_internal_state = get_internal_state


def main():
    from sglang.srt.managers.tokenizer_communicator_mixin import TokenizerCommunicatorMixin
    from sglang.launch_server import prepare_server_args, run_server, kill_process_tree
    install_metadata_readiness(TokenizerCommunicatorMixin)
    server_args = prepare_server_args(sys.argv[1:])
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == '__main__':
    main()
