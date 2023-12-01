"""Exceptions used across Keymaker"""


class AggregateException(Exception):
    """
    Exception type that aggregates multiple exceptions.

    This exception type can be constructed by appending other exceptions together.
    The individual exceptions are stored as a list and can be accessed using the 'exceptions' attribute.
    When the exception is raised, the concatenated string of exception messages is used as the error message.

    Example usage:
    try:
        # Some code that may raise exceptions
        raise ValueError("Invalid value")
    except Exception as e:
        aggregate_exception = AggregateException(e, TypeError("Type mismatch"))
        raise aggregate_exception
    """

    def __init__(self, *exceptions):
        self.exceptions = exceptions
        super().__init__(self._get_exception_messages())

    def _get_exception_messages(self):
        return "\n".join(str(exc) for exc in self.exceptions)


class Deprecation(Exception):
    """
    Exception for deprecations.
    """
