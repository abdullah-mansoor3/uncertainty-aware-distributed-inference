"""Tests for serial scheduler baseline."""

from src.scheduler.serial import SerialScheduler


def test_serial_scheduler_assigns_all_to_node_a() -> None:
    """Every subtask must route to node_a in serial baseline."""
    scheduler = SerialScheduler()
    subtasks = [
        {"id": 0, "text": "Task A", "dependencies": [], "parallel_safe": True},
        {"id": 1, "text": "Task B", "dependencies": [], "parallel_safe": True},
    ]
    scheduled = scheduler.schedule(subtasks)
    assert len(scheduled) == 2
    assert all(item["assigned_node"] == "node_a" for item in scheduled)
