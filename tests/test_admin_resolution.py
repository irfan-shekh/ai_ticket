"""
TicketAI - Admin AI & Manual Ticket Resolution Tests
Verifies:
1. Option A: AI-assisted resolution generation, editing, and approval (Status -> awaiting_customer).
2. Option B: Manual resolution authoring and dispatch (works independently of Gemini).
3. RBAC: Customers and unauthenticated users cannot access admin resolution actions.
4. Customer lifecycle: Customer confirms resolution (Status -> Resolved) or reports issue persists (reopened/needs_review).
5. Auditability: Resolution history preserves both AI and Manual records with admin identity.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app
from src.database.repositories import TicketRepository


class TestAdminResolution(unittest.TestCase):

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_rbac_non_admin_cannot_access_admin_resolution_endpoints(self):
        """Non-admin customer user must receive 401/403 on admin resolution routes."""
        with self.client.session_transaction() as sess:
            sess['user_id'] = 2
            sess['username'] = 'customer_user'
            sess['is_admin'] = False

        # Attempt to access manual resolve
        resp = self.client.post('/admin/resolve_manual', json={
            'ticket_id': 100,
            'summary': 'Hacked',
            'detailed_solution': 'I am customer pretending to be admin'
        })
        self.assertIn(resp.status_code, [401, 403])

        # Attempt to access AI generation
        resp2 = self.client.post('/admin/ticket/100/generate_ai_solution')
        self.assertIn(resp2.status_code, [401, 403])

        # Attempt to access AI approval
        resp3 = self.client.post('/admin/ticket/100/approve_ai_solution', json={
            'final_solution': 'Unauthorized approve'
        })
        self.assertIn(resp3.status_code, [401, 403])

        # Attempt to access resolution history
        resp4 = self.client.get('/admin/resolution_history')
        self.assertIn(resp4.status_code, [401, 403])

    def test_manual_resolution_independent_of_ai(self):
        """Admin manual resolution must succeed even if Gemini API is offline/unavailable."""
        mock_ticket = {
            'id': 101,
            'ticket_text': 'I cannot print my invoice receipt from dashboard.',
            'predicted_category': 'Billing',
            'notification_email': 'user@example.com',
            'user_email': 'user@example.com',
            'status': 'Open'
        }

        with patch.object(TicketRepository, 'get_ticket', return_value=mock_ticket), \
             patch.object(TicketRepository, 'save_manual_resolution', return_value=501) as mock_save, \
             patch('app.send_ticket_resolution_email', return_value=True):

            with self.client.session_transaction() as sess:
                sess['user_id'] = 1
                sess['username'] = 'admin'
                sess['is_admin'] = True

            payload = {
                'ticket_id': 101,
                'summary': 'Manual Invoice Re-send',
                'detailed_solution': 'We have generated your PDF receipt and attached it to your account.',
                'internal_notes': 'Verified transaction ID #TX992.',
                'resolution_category': 'Billing'
            }

            resp = self.client.post('/admin/resolve_manual', json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['status'], 'awaiting_customer')
            self.assertEqual(data['resolution_id'], 501)
            mock_save.assert_called_once_with(
                ticket_id=101,
                admin_id=1,
                summary='Manual Invoice Re-send',
                detailed_solution='We have generated your PDF receipt and attached it to your account.',
                internal_notes='Verified transaction ID #TX992.',
                resolution_category='Billing'
            )

    def test_admin_generate_ai_solution(self):
        """Admin requests AI proposed solution via Gemini + RAG on-demand."""
        mock_ticket = {
            'id': 102,
            'ticket_text': 'How do I return a damaged product within 30 days?',
            'predicted_category': 'Returns',
            'confidence': 96.5,
            'status': 'Open'
        }

        mock_rag_result = {
            'status': 'awaiting_customer',
            'resolution_data': {
                'problem_summary': 'Customer requests return steps for damaged product.',
                'identified_issue': 'Product return within policy window',
                'proposed_solution': [
                    'Locate original order receipt',
                    'Submit RMA request in customer portal',
                    'Pack item in original packaging with prepaid shipping label'
                ],
                'grounding_status': 'grounded',
                'requires_human_review': False,
                'escalation_reason': '',
                'model_name': 'gemini-3.8-flash'
            },
            'retrieved_documents': [
                {'doc_title': 'Returns Policy', 'relevance_score': 0.88}
            ]
        }

        with patch.object(TicketRepository, 'get_ticket', return_value=mock_ticket), \
             patch('src.resolution.resolution_engine.ResolutionEngine.process_ticket', return_value=mock_rag_result):

            with self.client.session_transaction() as sess:
                sess['user_id'] = 1
                sess['username'] = 'admin'
                sess['is_admin'] = True

            resp = self.client.post('/admin/ticket/102/generate_ai_solution')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertIn('proposed_solution', data)
            self.assertEqual(data['grounding_status'], 'grounded')
            self.assertEqual(len(data['sources']), 1)

    def test_admin_edit_and_approve_ai_solution(self):
        """Admin can edit AI solution, approve it, and dispatch to customer (Status -> awaiting_customer)."""
        mock_ticket = {
            'id': 103,
            'ticket_text': 'Need refund on double charge',
            'predicted_category': 'Billing',
            'notification_email': 'customer@example.com',
            'user_email': 'customer@example.com',
            'status': 'needs_review'
        }

        with patch.object(TicketRepository, 'get_ticket', return_value=mock_ticket), \
             patch.object(TicketRepository, 'approve_and_send_ai_resolution', return_value=True) as mock_approve, \
             patch('app.send_ticket_resolution_email', return_value=True):

            with self.client.session_transaction() as sess:
                sess['user_id'] = 1
                sess['username'] = 'admin'
                sess['is_admin'] = True

            payload = {
                'final_solution': '1. Verified double charge.\n2. Refund of $49.99 approved and scheduled within 3 business days.',
                'internal_notes': 'Approved refund code REF-881'
            }

            resp = self.client.post('/admin/ticket/103/approve_ai_solution', json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['status'], 'awaiting_customer')
            mock_approve.assert_called_once_with(
                ticket_id=103,
                admin_id=1,
                final_solution=payload['final_solution'],
                internal_notes='Approved refund code REF-881'
            )

    def test_customer_confirms_resolution(self):
        """Customer marks issue as resolved (Status -> Resolved)."""
        mock_ticket = {'id': 104, 'user_id': 2}
        with patch.object(TicketRepository, 'get_ticket', return_value=mock_ticket), \
             patch.object(TicketRepository, 'get_latest_resolution', return_value={'id': 20}), \
             patch.object(TicketRepository, 'save_feedback', return_value=1) as mock_fb:

            with self.client.session_transaction() as sess:
                sess['user_id'] = 2
                sess['username'] = 'customer_user'
                sess['is_admin'] = False

            resp = self.client.post('/api/ticket/104/feedback', json={
                'was_resolved': True,
                'rating': 5,
                'comment': 'Awesome, problem completely fixed!'
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertTrue(data['was_resolved'])
            mock_fb.assert_called_once_with(
                ticket_id=104,
                resolution_id=20,
                rating=5,
                comment='Awesome, problem completely fixed!',
                was_resolved=True
            )

    def test_customer_reopens_ticket(self):
        """Customer reopens ticket if issue persists (Status -> reopened)."""
        mock_ticket = {'id': 105, 'user_id': 2, 'status': 'awaiting_customer'}
        with patch.object(TicketRepository, 'get_ticket', return_value=mock_ticket), \
             patch.object(TicketRepository, 'reopen_ticket', return_value=True) as mock_reopen:

            with self.client.session_transaction() as sess:
                sess['user_id'] = 2
                sess['username'] = 'customer_user'
                sess['is_admin'] = False

            resp = self.client.post('/api/ticket/105/reopen', json={
                'reason': 'The login error still happens in Safari.'
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['status'], 'reopened')
            mock_reopen.assert_called_once_with(
                105,
                reason='The login error still happens in Safari.',
                user_id=2
            )

    def test_resolution_history_api(self):
        """Admin can fetch complete audit history of AI and Manual resolutions."""
        mock_history = [
            {
                'resolution_id': 1,
                'ticket_id': 101,
                'resolution_method': 'MANUAL',
                'admin_username': 'admin',
                'proposed_solution': 'Manual steps',
                'final_solution': 'Manual steps',
                'current_ticket_status': 'awaiting_customer'
            },
            {
                'resolution_id': 2,
                'ticket_id': 102,
                'resolution_method': 'AI',
                'admin_username': 'admin',
                'proposed_solution': 'AI proposed steps',
                'final_solution': 'Admin edited AI steps',
                'current_ticket_status': 'Resolved'
            }
        ]

        with patch.object(TicketRepository, 'list_resolution_history', return_value=mock_history):
            with self.client.session_transaction() as sess:
                sess['user_id'] = 1
                sess['is_admin'] = True

            resp = self.client.get('/admin/resolution_history')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['count'], 2)
            self.assertEqual(data['history'][0]['resolution_method'], 'MANUAL')
            self.assertEqual(data['history'][1]['resolution_method'], 'AI')

    def test_admin_metrics_comparison(self):
        """Admin metrics accurately compare AI-Assisted and Manual resolution counts."""
        mock_metrics = {
            'total_tickets': 20,
            'open_tickets': 5,
            'pending_tickets': 3,
            'awaiting_customer': 4,
            'resolved_tickets': 8,
            'reopened_tickets': 2,
            'escalated_tickets': 3,
            'ai_assisted_resolutions': 12,
            'manual_resolutions': 5,
            'customer_satisfaction': 4.75,
            'resolution_rate': 40.0,
            'resolution_comparison': {'ai': 12, 'manual': 5}
        }

        with patch.object(TicketRepository, 'get_admin_metrics', return_value=mock_metrics):
            with self.client.session_transaction() as sess:
                sess['user_id'] = 1
                sess['is_admin'] = True

            resp = self.client.get('/admin/analytics')
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertTrue(data['success'])
            m = data['metrics']
            self.assertEqual(m['ai_assisted_resolutions'], 12)
            self.assertEqual(m['manual_resolutions'], 5)
            self.assertEqual(m['resolution_comparison']['ai'], 12)
            self.assertEqual(m['resolution_comparison']['manual'], 5)


if __name__ == '__main__':
    unittest.main()

