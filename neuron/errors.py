class ImproperDateFormatError(Exception):
    """
    Error class for values that cannot be parsed by datetime.date's constructors.
    """

    pass


class ImproperGRChVersionError(Exception):
    """
    Error class for invalid GRCh Version values.
    """

    pass


class InvalidDateRangeError(Exception):
    """Execption for unexpected date ranges"""

    pass


class InvalidArgumentSetError(Exception):
    """
    Error class for function/method arguments that cannot be used together
    """

    pass


class MSAArrayFileNotFound(Exception):
    """
    MSA .npy file was required but was not found
    """

    pass


class UnexpectedMutationError(Exception):
    """
    Error class for incomplete or invalid HGVS-style mutation codes
    """

    pass


class UnexpectedResidueError(Exception):
    """
    Error class for invalid sequences
    """

    pass


class SequenceNotFoundError(Exception):
    """
    Error class for not found
    """

    pass


class InvalidArgumentTypeError(Exception):
    """
    Error class for invalid argument type
    """

    pass


class UnexpectedColumnNameError(Exception):
    """
    Error class for Unexpected Column Name
    """

    pass


class NoActionPerformedWarning(RuntimeWarning):
    """
    Raised when a function was called but no changes occurred due to one or more function arguments.
    """

    pass


class EmptySearchBufferError(Exception):
    """
    Raised when searcing empty buffer.
    """

    pass
