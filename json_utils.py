from dataclasses import fields
from typing import Any, Dict, Type, get_origin, get_args, get_type_hints, TypeVar, Generic
import json
import os
import numpy as np
import torch

T = TypeVar('T', bound='JsonSerializationMixin')
class JsonSerializationMixin(Generic[T]):
    def to_dict(self) -> Dict[str, Any]:
        data = {}
        for field_ in fields(self):
            if not field_.repr:  # Skip fields where init=False
                continue
            value = getattr(self, field_.name)
            if isinstance(value, np.ndarray):
                data[field_.name] = value.tolist()
            elif isinstance(value, torch.Tensor):
                data[field_.name] = value.cpu().detach().numpy().tolist()
            elif isinstance(value, list):
                # Handle lists of dataclasses or other serializable types
                data[field_.name] = [item.to_dict() if hasattr(item, 'to_dict') else item for item in value]
            elif isinstance(value, dict): # Handle dictionaries
                # Check if dictionary values are serializable
                first_value = next(iter(value.values()), None) # Peek at the first value to check type
                if first_value and hasattr(first_value, 'to_dict'):
                    data[field_.name] = {k: v.to_dict() for k, v in value.items()} # Serialize dict values
                else:
                    data[field_.name] = value # Otherwise, keep as is
            elif hasattr(value, 'to_dict'): # Check if value itself has to_dict (for nested dataclasses)
                data[field_.name] = value.to_dict()
            else:
                data[field_.name] = value
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        type_hints = get_type_hints(cls)
        kwargs = {}
        dataclass_fields = {f.name for f in fields(cls)}
        for name, value in data.items():
            if name in dataclass_fields:
                if name in type_hints:
                    origin = get_origin(type_hints[name]) # Use get_origin to get the base type
                    args = get_args(type_hints[name])     # Use get_args to get type arguments

                    if isinstance(value, list) and type_hints[name] == np.ndarray:
                        kwargs[name] = np.array(value)
                    elif isinstance(value, list) and type_hints[name] == torch.Tensor:
                        kwargs[name] = torch.tensor(value)
                    elif isinstance(value, list) and origin is list and args and hasattr(args[0], 'from_dict'):
                        # Handle lists of dataclasses
                        item_type = args[0]
                        kwargs[name] = [item_type.from_dict(item_data) for item_data in value]
                    elif origin is dict and args and hasattr(args[1], 'from_dict') and isinstance(value, dict):
                        # Handle dictionaries of dataclasses
                        value_type = args[1] # Get the value type from Dict[KeyType, ValueType]
                        kwargs[name] = {k: value_type.from_dict(v) for k, v in value.items()}
                    elif hasattr(type_hints[name], 'from_dict') and isinstance(value, dict):
                        # Handle nested dataclasses
                        kwargs[name] = type_hints[name].from_dict(value)
                    else:
                        kwargs[name] = value
                else:
                    kwargs[name] = value # For attributes not in type hints, pass as is
        return cls(**kwargs)

    def save_json(self, filepath: str) -> None:
        """Saves the object to a JSON file."""
        # if filepath is just a name
        if os.path.dirname(filepath):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        with open(filepath, 'w') as f: # Open in text mode 'w' for JSON
            json.dump(self.to_dict(), f, indent=4) # Use json.dump and to_dict

    @classmethod
    def load_json(cls: Type[T], filepath: str) -> T:
        """Loads the object from a JSON file."""
        with open(filepath, 'r') as f: # Open in text mode 'r' for JSON
            data = json.load(f) # Use json.load
        return cls.from_dict(data) # Use cls.from_dict to reconstruct