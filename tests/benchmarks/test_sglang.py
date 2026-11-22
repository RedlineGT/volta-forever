"""SGLang exposes request counters once and forward timers on every TP rank."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

DIR = Path(__file__).resolve().parent
CLIENT = DIR.parents[1] / "docs/guides/benchmarks/sglang.py"
spec = importlib.util.spec_from_file_location("sglang_benchmark", CLIENT)
sglang = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sglang)


def timer(rank, phase, seconds):
    return f'sglang:forward_execution_seconds_total{{model_name="qwen",tp_rank="{rank}",category="{phase}"}} {seconds}'


def request_tokens():
    return ['sglang:prefill_effective_tokens_total{model_name="qwen",tp_rank="0",mode="input"} 227',
            'sglang:prompt_tokens_total{model_name="qwen"} 227',
            'sglang:generation_tokens_total{model_name="qwen"} 64']


class RankTimers(unittest.TestCase):
    def measured(self, before, after, tp):
        with patch.object(sglang, "counters", return_value=after), patch.object(sglang.time, "sleep"):
            return sglang.phase_seconds("unused", "qwen", before, 227, 64, tp)

    def test_real_two_rank_scrapes(self):
        def read(name):
            lines = (DIR / "fixtures" / f"sglang-tp2-{name}.prom").read_text().splitlines()
            return sglang.parse_counters(lines, "qwen", 2)
        a, b = read("before"), read("after")
        measured = self.measured(a, b, 2)
        self.assertEqual([r["tp_rank"] for r in measured], [0, 1])
        for rank, row in enumerate(measured):
            for phase, field in (("extend", "prefill_s"), ("decode", "decode_s")):
                key = (phase, rank)
                self.assertEqual(row[field], b[key] - a[key])
                self.assertGreater(row[field], 0)
        self.assertGreater(sum(r["decode_s"] for r in measured), max(r["decode_s"] for r in measured))

    def test_one_and_four_rank_accounting(self):
        for tp in (1, 4):
            with self.subTest(tp=tp):
                lines = request_tokens()[1:]
                for rank in range(tp):
                    for mode in ("input", *sglang.CACHE_HITS):
                        count = 227 if rank == 0 and mode == "input" else 0
                        lines.append(f'sglang:prefill_effective_tokens_total{{model_name="qwen",tp_rank="{rank}",mode="{mode}"}} {count}')
                    lines.extend([timer(rank, "extend", 1 + rank), timer(rank, "decode", 5 - rank)])
                rows = self.measured({}, sglang.parse_counters(lines, "qwen", tp), tp)
                self.assertEqual([r["prefill_s"] for r in rows], list(range(1, tp + 1)))
                self.assertEqual([r["decode_s"] for r in rows], list(range(5, 5 - tp, -1)))

    def test_request_uses_slowest_rank_in_each_phase(self):
        usage = {"prompt_tokens": 227, "completion_tokens": 64}
        events = [{"choices": [{"delta": {"content": "Kyoto"}}], "usage": usage}]
        timings = [{"tp_rank": 0, "prefill_s": 0.08, "decode_s": 0.73},
                   {"tp_rank": 1, "prefill_s": 0.09, "decode_s": 0.71}]
        for tp in (1, 2):
            with self.subTest(tp=tp), patch.object(sglang, "urlopen"), \
                    patch.object(sglang, "stream_events", return_value=events), \
                    patch.object(sglang, "counters"), \
                    patch.object(sglang, "phase_seconds", return_value=timings[:tp]):
                row = sglang.measure("http://localhost", "qwen", {"id": "run1", "city": "Kyoto", "prompt": "test"}, tp)
                self.assertEqual(row["prefill_tokens_per_s"], 227 / (0.08 if tp == 1 else 0.09))
                self.assertEqual(row["decode_tokens_per_s"], 63 / 0.73)
                self.assertEqual("gpu_rank_phases" in row, tp > 1)
                self.assertEqual("client_ttft_s" in row, tp > 1)

    def test_rejects_duplicate_or_incomplete_rank_sets(self):
        lines = [timer(i, "extend", i + 1) for i in range(4)]
        for bad in (lines[:-1], lines + [lines[0]], lines[:-1] + [timer(4, "extend", 2)],
                    [line.replace('tp_rank="0",', '') for line in lines]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sglang.parse_counters(bad, "qwen", 4)

    def test_rejects_speculative_phases_and_resets(self):
        for a, b in (({}, [timer(0, "target_verify", 1)]), ({("extend", 0): 2}, [timer(0, "extend", 1)])):
            with self.subTest(a=a,b=b), self.assertRaises(ValueError):
                self.measured(a, sglang.parse_counters(b, "qwen", 1), 1)


if __name__ == "__main__":
    unittest.main()
