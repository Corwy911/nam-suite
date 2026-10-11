import abc as _abc
from typing import Any as _Any
from typing import Dict as _Dict


class ExportModelDictPostHook(_abc.ABC):
    """
    Hook run after a model export dictionary has been assembled.
    """

    @_abc.abstractmethod
    def apply(self, model_dict: _Dict[str, _Any]) -> _Dict[str, _Any]: ...
