from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Type, Union

from psiking.core.base.component import BaseComponent

class BaseReader(BaseComponent):
    """The base class for all readers"""

    _dependencies = []
    _name="BaseReader"
    
    @property
    def default_metadata(self):
        return {
            'reader': self._name
        }
    
    ...