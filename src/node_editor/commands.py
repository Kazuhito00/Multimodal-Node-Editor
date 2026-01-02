from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .models import Node, Connection
    from .core import Graph


class Command(ABC):
    """Base class for all commands in the undo/redo framework."""
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass


class CommandHistory:
    """Manages the execution, undoing, and redoing of commands."""
    def __init__(self):
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []

    def execute(self, command: Command):
        """Executes a command and adds it to the undo stack."""
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()

    def undo(self):
        """Undoes the most recent command."""
        if not self._undo_stack:
            return
        command = self._undo_stack.pop()
        command.undo()
        self._redo_stack.append(command)

    def redo(self):
        """Redoes the most recently undone command."""
        if not self._redo_stack:
            return
        command = self._redo_stack.pop()
        command.execute()
        self._undo_stack.append(command)


class AddNodeCommand(Command):
    """Command to add a node to the graph."""
    def __init__(self, graph: 'Graph', node: 'Node'):
        self.graph = graph
        self.node = node

    def execute(self):
        self.graph.nodes.append(self.node)

    def undo(self):
        self.graph.nodes.remove(self.node)


class RemoveNodeCommand(Command):
    """Command to remove a node and its connections from the graph."""
    def __init__(self, graph: 'Graph', node: 'Node'):
        self.graph = graph
        self.node = node
        self.removed_connections: List['Connection'] = []

    def execute(self):
        self.removed_connections = [
            c for c in self.graph.connections
            if c.from_node_id == self.node.id or c.to_node_id == self.node.id
        ]
        self.graph.connections = [
            c for c in self.graph.connections
            if c.from_node_id != self.node.id and c.to_node_id != self.node.id
        ]
        self.graph.nodes.remove(self.node)

    def undo(self):
        self.graph.nodes.append(self.node)
        self.graph.connections.extend(self.removed_connections)
        self.removed_connections = []


class AddConnectionCommand(Command):
    """Command to add a connection to the graph."""
    def __init__(self, graph: 'Graph', connection: 'Connection'):
        self.graph = graph
        self.connection = connection

    def execute(self):
        self.graph.connections.append(self.connection)

    def undo(self):
        self.graph.connections.remove(self.connection)