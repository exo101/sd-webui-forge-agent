class H3StudioError(RuntimeError):
    """Expected user-facing error."""


class BackendUnavailable(H3StudioError):
    """The configured ComfyUI backend cannot be reached."""


class WorkflowValidationError(H3StudioError):
    """The requested H3 job cannot be represented safely."""

