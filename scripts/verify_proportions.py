"""Verify that the Premium segment has significant proportion test results"""
import pandas as pd

df = pd.read_csv('data/sample_ab_data.csv')

print("=" * 60)
print("CONVERSION RATE VERIFICATION")
print("=" * 60)

for segment in df['customer_segment'].unique():
    segment_data = df[df['customer_segment'] == segment]
    print(f"\n{segment} Segment:")
    print(f"  Total: {len(segment_data)}")

    for group in ['control', 'treatment']:
        group_data = segment_data[segment_data['experiment_group'] == group]
        conversions = (group_data['post_effect'] > 0).sum()
        total = len(group_data)
        rate = conversions / total if total > 0 else 0
        print(f"  {group}: {conversions}/{total} = {rate:.1%}")

    # Calculate difference
    control_data = segment_data[segment_data['experiment_group'] == 'control']
    treatment_data = segment_data[segment_data['experiment_group'] == 'treatment']

    control_rate = (control_data['post_effect'] > 0).sum() / len(control_data) if len(control_data) > 0 else 0
    treatment_rate = (treatment_data['post_effect'] > 0).sum() / len(treatment_data) if len(treatment_data) > 0 else 0

    diff = treatment_rate - control_rate
    print(f"  Difference: {diff:+.1%}")

print("\n" + "=" * 60)
print("Expected: Premium segment should have ~23% point difference")
print("(Control ~45%, Treatment ~68%)")
print("=" * 60)
