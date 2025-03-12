import unittest
import json
import os
import tempfile
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional

# Assuming your json_utils.py is in the same directory, otherwise adjust the import
from json_utils import Serial  # Replace 'your_module' with the actual module name


# Define some dataclasses for testing
@dataclass
class NestedData(Serial):
    value: int = 10
    text: str = "nested"


@dataclass
class DataDictionary(Serial):
    name: str = "dict_example"
    data_dict: dict[str, NestedData] = field(default_factory=dict) # Dictionary of NestedData


@dataclass
class SampleData(Serial):
    name: str = "example"
    count: int = 42
    data_list: List[int] = field(default_factory=lambda: [1, 2, 3])
    numpy_array: np.ndarray = field(default_factory=lambda: np.array([4, 5, 6]))
    torch_tensor: torch.Tensor = field(default_factory=lambda: torch.tensor([7, 8, 9]))
    nested: NestedData = field(default_factory=NestedData)
    optional_value: Optional[int] = None
    not_initialized: str = field(default="default_value", repr=False) # Example of field with init=False


class TestJsonSerializationMixin(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_filepath = os.path.join(self.temp_dir.name, "test_data.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_to_dict(self):
        sample_instance = SampleData()
        data_dict = sample_instance.to_dict()

        expected_dict = {
            'name': 'example',
            'count': 42,
            'data_list': [1, 2, 3],
            'numpy_array': [4, 5, 6],
            'torch_tensor': [7, 8, 9],
            'nested': {'value': 10, 'text': 'nested'},
            'optional_value': None,
        }
        self.assertEqual(data_dict, expected_dict)
        self.assertNotIn('not_initialized', data_dict) # Check field with init=False is skipped

    def test_from_dict(self):
        data_dict = {
            'name': 'test_name',
            'count': 100,
            'data_list': [4, 5, 6],
            'numpy_array': [1, 2, 3],
            'torch_tensor': [9, 8, 7],
            'nested': {'value': 20, 'text': 'another_nested'},
            'optional_value': 123,
            'extra_field': 'should_be_ignored_in_dataclass' # Extra field, should be ignored by from_dict in strict mode
        }
        reconstructed_instance = SampleData.from_dict(data_dict)

        self.assertEqual(reconstructed_instance.name, 'test_name')
        self.assertEqual(reconstructed_instance.count, 100)
        self.assertEqual(reconstructed_instance.data_list, [4, 5, 6])
        np.testing.assert_array_equal(reconstructed_instance.numpy_array, np.array([1, 2, 3]))
        torch.testing.assert_close(reconstructed_instance.torch_tensor, torch.tensor([9, 8, 7])) # Use torch.testing.assert_close for tensors
        self.assertEqual(reconstructed_instance.nested.value, 20)
        self.assertEqual(reconstructed_instance.nested.text, 'another_nested')
        self.assertEqual(reconstructed_instance.optional_value, 123)
        self.assertEqual(reconstructed_instance.not_initialized, "default_value") # Check default value for init=False fields


    def test_save_json(self):
        sample_instance = SampleData(name="save_test", count=50)
        sample_instance.save_json(self.temp_filepath)

        self.assertTrue(os.path.exists(self.temp_filepath))
        with open(self.temp_filepath, 'r') as f:
            loaded_data = json.load(f)

        expected_data = {
            'name': 'save_test',
            'count': 50,
            'data_list': [1, 2, 3],
            'numpy_array': [4, 5, 6],
            'torch_tensor': [7, 8, 9],
            'nested': {'value': 10, 'text': 'nested'},
            'optional_value': None,
        } # Note: original numpy/torch data is used in default sample instance, not 'save_test' instance
        self.assertEqual(loaded_data, expected_data)


    def test_load_json(self):
        # Create a json file to load from
        test_json_data = {
            'name': 'load_test',
            'count': 75,
            'data_list': [7, 8, 9],
            'numpy_array': [10, 11, 12],
            'torch_tensor': [13, 14, 15],
            'nested': {'value': 30, 'text': 'loaded_nested'},
            'optional_value': 456,
        }
        with open(self.temp_filepath, 'w') as f:
            json.dump(test_json_data, f)

        loaded_instance = SampleData.load_json(self.temp_filepath)

        self.assertEqual(loaded_instance.name, 'load_test')
        self.assertEqual(loaded_instance.count, 75)
        self.assertEqual(loaded_instance.data_list, [7, 8, 9])
        np.testing.assert_array_equal(loaded_instance.numpy_array, np.array([10, 11, 12]))
        torch.testing.assert_close(loaded_instance.torch_tensor, torch.tensor([13, 14, 15]))
        self.assertEqual(loaded_instance.nested.value, 30)
        self.assertEqual(loaded_instance.nested.text, 'loaded_nested')
        self.assertEqual(loaded_instance.optional_value, 456)
        self.assertEqual(loaded_instance.not_initialized, "default_value") # Check default value for init=False fields


    def test_to_dict_with_dict_of_classes(self):
        nested1 = NestedData(value=100, text="dict_nested_1")
        nested2 = NestedData(value=200, text="dict_nested_2")
        dict_instance = DataDictionary(data_dict={"item1": nested1, "item2": nested2})
        data_dict = dict_instance.to_dict()

        expected_dict = {
            'name': 'dict_example',
            'data_dict': {
                'item1': {'value': 100, 'text': 'dict_nested_1'},
                'item2': {'value': 200, 'text': 'dict_nested_2'}
            }
        }
        self.assertEqual(data_dict, expected_dict)

    def test_from_dict_with_dict_of_classes(self):
        data_dict_input = {
            'name': 'dict_from_dict_test',
            'data_dict': {
                'key1': {'value': 300, 'text': 'from_dict_nested_1'},
                'key2': {'value': 400, 'text': 'from_dict_nested_2'}
            }
        }
        dict_instance = DataDictionary.from_dict(data_dict_input)

        self.assertEqual(dict_instance.name, 'dict_from_dict_test')
        self.assertIsInstance(dict_instance.data_dict, dict)
        self.assertIsInstance(dict_instance.data_dict['key1'], NestedData)
        self.assertEqual(dict_instance.data_dict['key1'].value, 300)
        self.assertEqual(dict_instance.data_dict['key1'].text, 'from_dict_nested_1')
        self.assertEqual(dict_instance.data_dict['key2'].value, 400)
        self.assertEqual(dict_instance.data_dict['key2'].text, 'from_dict_nested_2')

    def test_save_load_json_with_dict_of_classes(self):
        nested1 = NestedData(value=500, text="save_load_nested_1")
        nested2 = NestedData(value=600, text="save_load_nested_2")
        dict_instance = DataDictionary(name="dict_save_load_test", data_dict={"a": nested1, "b": nested2})

        dict_instance.save_json(self.temp_filepath)
        loaded_instance = DataDictionary.load_json(self.temp_filepath)

        self.assertEqual(loaded_instance.name, "dict_save_load_test")
        self.assertIsInstance(loaded_instance.data_dict, dict)
        self.assertIsInstance(loaded_instance.data_dict['a'], NestedData)
        self.assertEqual(loaded_instance.data_dict['a'].value, 500)
        self.assertEqual(loaded_instance.data_dict['b'].text, 'save_load_nested_2')
        # Check other loaded values as needed


if __name__ == '__main__':
    unittest.main()
