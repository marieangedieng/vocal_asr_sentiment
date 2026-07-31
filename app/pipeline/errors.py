class PipelineError(Exception):
    """Base class for user-facing pipeline errors."""

    status_code = 400


class UnsupportedAudioFormatError(PipelineError):
    status_code = 415


class EmptyAudioError(PipelineError):
    status_code = 400


class CorruptedAudioError(PipelineError):
    status_code = 400


class SilentAudioError(PipelineError):
    status_code = 422


class AudioTooLongError(PipelineError):
    status_code = 413

