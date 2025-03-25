class ActionHistory:
    def __init__(self):
        self._id = 0  # Counter for the current ID of objects
        self.undo_stack = {}  # Stack to keep track of undo actions
        self.redo_stack = {}  # Stack to keep track of redo actions

    def add_action(self, object_on_stack):
        """
        Add a new action to the history. This clears the redo stack.
        """
        self.undo_stack[self._id] = object_on_stack

    def undo(self):
        """
        Undo the last action.
        Returns the row_id and actors for the undone action.
        """
        if self._id - 1 in self.undo_stack:
            self._id -= 1
            action = self.undo_stack.pop(self._id)
            self.redo_stack[self._id] = action
            return action
        return None

    def redo(self):
        """
        Redo the last undone action.
        Returns the row_id and actors for the redone action.
        """
        if self._id in self.redo_stack:
            action = self.redo_stack.pop(self._id)
            self.undo_stack[self._id] = action
            self._id += 1
            return action
        return None

    def remove_by_id(self, id: int):
        """
        Remove action by ID from both undo and redo stacks.
        """
        if id in self.undo_stack:
            del self.undo_stack[id]
        if id in self.redo_stack:
            del self.redo_stack[id]

    def get_id(self):
        return self._id

    def decrementIndex(self):
        if self._id > 0:
            self._id -= 1

    def incrementIndex(self):
        self._id += 1

    def clearIndex(self):
        self._id = 0
