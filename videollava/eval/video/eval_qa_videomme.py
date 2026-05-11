import json
import re
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', required=True, help='Path to the inference output .json file')
    return parser.parse_args()


def extract_letter(text):
    """Extract A/B/C/D from model output. Tries strict match first, then first letter fallback."""
    text = text.strip().upper()
    # Prefer explicit single letter at start of output
    match = re.match(r'^([ABCD])\b', text)
    if match:
        return match.group(1)
    # Fallback: find first occurrence of A/B/C/D
    match = re.search(r'\b([ABCD])\b', text)
    if match:
        return match.group(1)
    return None


def evaluate(pred_file):
    results = [json.loads(line) for line in open(pred_file) if line.strip()]

    total_correct = 0
    total_count = 0

    by_duration = defaultdict(lambda: {'correct': 0, 'total': 0})

    for r in results:
        pred_letter = extract_letter(r['pred'])
        gt_letter = r['answer'].strip().upper()
        duration = r.get('duration', 'unknown')

        is_correct = (pred_letter == gt_letter)

        total_correct += int(is_correct)
        total_count += 1
        by_duration[duration]['correct'] += int(is_correct)
        by_duration[duration]['total'] += 1

    print("=" * 40)
    print("VideoMME Evaluation Results")
    print("=" * 40)

    for dur in ['short', 'medium', 'long', 'unknown']:
        if dur in by_duration:
            d = by_duration[dur]
            acc = d['correct'] / d['total'] * 100
            print(f"  {dur.capitalize():8s}: {d['correct']:4d}/{d['total']:4d} = {acc:.2f}%")

    print("-" * 40)
    if total_count > 0:
        overall_acc = total_correct / total_count * 100
        print(f"  {'Overall':8s}: {total_correct:4d}/{total_count:4d} = {overall_acc:.2f}%")
    print("=" * 40)

    # Count unparseable predictions
    unparseable = sum(1 for r in results if extract_letter(r['pred']) is None)
    if unparseable:
        print(f"  [Warning] {unparseable}/{total_count} predictions could not be parsed as A/B/C/D")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.pred_file)
