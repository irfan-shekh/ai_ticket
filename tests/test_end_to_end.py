"""
TicketAI - End-to-End Pipeline Integration Tests
Verifies the complete flow from submission to verified resolution.
Supports mock execution so unit test suites run without external network or API keys.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.services.ticket_orchestrator import TicketOrchestrator
from src.database.repositories import TicketRepository


class TestEndToEndPipeline(unittest.TestCase):

    @patch("src.services.ticket_orchestrator.get_db")
    @patch("src.resolution.resolution_engine.TicketRepository")
    def test_mocked_end_to_end_ticket_processing(self, mock_repo, mock_get_db):
        """Simulate complete ticket submission workflow with mocked database."""
        # Setup mock DB cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 999
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.execute.return_value = mock_cursor
        mock_get_db.return_value.__enter__.return_value = mock_conn

        mock_repo.save_resolution.return_value = 888
        mock_repo.update_status.return_value = True

        orchestrator = TicketOrchestrator()
        result = orchestrator.process_ticket_submission(
            user_id=1,
            ticket_text="My laptop screen went dark and cannot boot up into Windows.",
            notification_email="user@example.com"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["ticket_id"], 999)
        self.assertEqual(result["raw_category"], "Technical Support")
        self.assertGreaterEqual(result["confidence"], 60.0)
        self.assertIn("resolution", result)
        self.assertIn("status", result)


if __name__ == "__main__":
    unittest.main()

