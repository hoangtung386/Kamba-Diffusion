import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.registry import Registry

class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = Registry('test_registry')
        
    def test_register_and_get(self):
        @self.registry.register('item1')
        class Item1:
            pass
            
        cls = self.registry.get('item1')
        self.assertEqual(cls, Item1)
        
    def test_key_error(self):
        with self.assertRaises(KeyError):
            self.registry.get('non_existent')
            
    def test_list(self):
        @self.registry.register('a')
        class A: pass
        @self.registry.register('b')
        class B: pass
        
        keys = self.registry.list()
        self.assertIn('a', keys)
        self.assertIn('b', keys)

if __name__ == '__main__':
    unittest.main()
