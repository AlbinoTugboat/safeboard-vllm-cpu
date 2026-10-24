import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from soak_test import summarize


class SustainedSummaryTests(unittest.TestCase):
    def test_failures_count_and_do_not_contribute_tokens_or_latency(self):
        rows = [{'success': True, 'prompt_tokens': 40, 'output_tokens': 64, 'ttft_s': 1, 'latency_s': 5, 'tpot_s': 0.1},
                {'success': False, 'error_type': 'TimeoutError'},
                {'success': True, 'prompt_tokens': 1700, 'output_tokens': 128, 'ttft_s': 3, 'latency_s': 9, 'tpot_s': 0.2}]
        result = summarize(rows, 12)
        self.assertEqual((result['requests'], result['successful'], result['failed']), (3, 2, 1))
        self.assertEqual(result['output_tokens'], 192)
        self.assertEqual(result['output_tokens_per_s'], 16)
        self.assertEqual(result['ttft_s']['mean'], 2)
        self.assertEqual(result['latency_s']['p95'], 9)
        self.assertEqual(result['prompt_tokens'], {'mean': 870, 'min': 40, 'max': 1700})

    def test_all_failed_run_has_no_fabricated_latency(self):
        result = summarize([{'success': False}], 10)
        self.assertEqual(result['failed'], 1)
        self.assertEqual(result['output_tokens_per_s'], 0)
        self.assertNotIn('latency_s', result)

    def test_empty_window_cannot_divide_by_zero(self):
        self.assertEqual(summarize([], 0)['output_tokens_per_s'], 0)


if __name__ == '__main__':
    unittest.main()
