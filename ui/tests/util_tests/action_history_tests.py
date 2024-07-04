import unittest
from PyQt5.QtCore import QObject
from util.action_history import ActionHistory

class ActionHistoryTests(unittest.TestCase):

    def setUp(self):
        self.parent = QObject()
        self.history = ActionHistory()
    
    def test_add_action_single(self):
        action = {'action': 'add', 'details': 'test action'}
        self.history.add_action(action)
        self.assertIn(0, self.history.undo_stack, f"Failed in test_add_action_single, undo_stack: {self.history.undo_stack}")
        self.assertEqual(self.history.undo_stack[0], action, f"Failed in test_add_action_single, expected action: {action}, actual action: {self.history.undo_stack.get(0)}")

    def test_add_action_multiple(self):
        actions = [{'action': f'add_{i}', 'details': f'test action {i}'} for i in range(5)]
        for i, action in enumerate(actions):
            self.history.add_action(action)
            self.history.incrementIndex()
            self.assertIn(i, self.history.undo_stack, f"Failed in test_add_action_multiple with action id: {i}, undo_stack: {self.history.undo_stack}")
            self.assertEqual(self.history.undo_stack[i], action, f"Failed in test_add_action_multiple with action id: {i}, expected action: {action}, actual action: {self.history.undo_stack.get(i)}")
    
    def test_undo(self):
        actions = [{'action': f'add_{i}', 'details': f'test action {i}'} for i in range(5)]
        for i, action in enumerate(actions):
            self.history.add_action(action)
            self.history.incrementIndex()

        for i in range(4, -1, -1):
            undone_action = self.history.undo()
            self.assertEqual(undone_action, actions[i], f"Failed in test_undo with action id: {i}, expected undone action: {actions[i]}, actual undone action: {undone_action}")
            self.assertIn(i, self.history.redo_stack, f"Failed in test_undo with action id: {i}, redo_stack: {self.history.redo_stack}")

    def test_redo(self):
        actions = [{'action': f'add_{i}', 'details': f'test action {i}'} for i in range(5)]
        for i, action in enumerate(actions):
            self.history.add_action(action)
            self.history.incrementIndex()

        for i in range(4, -1, -1):
            undone_action = self.history.undo()

        for i, action in enumerate(actions):
            redone_action = self.history.redo()
            self.assertEqual(redone_action, action, f"Failed in test_redo with action id: {i}, expected redone action: {action}, actual redone action: {redone_action}")
            self.assertIn(i, self.history.undo_stack, f"Failed in test_redo with action id: {i}, undo_stack: {self.history.undo_stack}")

    def test_remove_by_id(self):
        actions = [{'action': f'add_{i}', 'details': f'test action {i}'} for i in range(5)]
        for i, action in enumerate(actions):
            self.history.add_action(action)

        for i in range(5):
            self.history.remove_by_id(i)
            self.assertNotIn(i, self.history.undo_stack, f"Failed in test_remove_by_id with action id: {i}, undo_stack: {self.history.undo_stack}")

    def test_increment_and_decrement_index(self):
        for i in range(5):
            self.history.incrementIndex()
            self.assertEqual(self.history.get_id(), i + 1, f"Failed in test_increment_and_decrement_index during increment with expected id: {i + 1}, actual id: {self.history.get_id()}")
        for i in range(5):
            self.history.decrementIndex()
            self.assertEqual(self.history.get_id(), 4 - i, f"Failed in test_increment_and_decrement_index during decrement with expected id: {4 - i}, actual id: {self.history.get_id()}")

    def test_clear_index(self):
        for i in range(5):
            self.history.incrementIndex()
        self.history.clearIndex()
        self.assertEqual(self.history.get_id(), 0, f"Failed in test_clear_index with expected id: 0, actual id: {self.history.get_id()}")


if __name__ == '__main__':
    unittest.main()
