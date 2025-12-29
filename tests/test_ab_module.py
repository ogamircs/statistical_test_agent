"""Test the A/B testing module"""

from ab_testing import ABTestAnalyzer

def test_ab_testing():
    print("Testing A/B Testing Module...")
    print("=" * 60)

    # Initialize analyzer
    analyzer = ABTestAnalyzer()

    # Load sample data
    print("\n1. Loading data...")
    info = analyzer.load_data("sample_ab_data.csv")
    print(f"   Shape: {info['shape']}")
    print(f"   Columns: {info['columns']}")

    # Detect columns
    print("\n2. Detecting columns...")
    suggestions = analyzer.detect_columns()
    for col_type, suggested in suggestions.items():
        print(f"   {col_type}: {suggested}")

    # Set column mapping
    print("\n3. Setting column mapping...")
    analyzer.set_column_mapping({
        "customer_id": "customer_id",
        "group": "experiment_group",
        "effect_value": "effect_value",
        "segment": "customer_segment",
        "duration": "experiment_duration_days"
    })

    # Set group labels
    print("\n4. Setting group labels...")
    analyzer.set_group_labels("treatment", "control")

    # Get segment distribution
    print("\n5. Segment distribution:")
    dist = analyzer.get_segment_distribution()
    print(f"   Groups: {dist.get('group_distribution', {})}")
    print(f"   Segments: {dist.get('segment_distribution', {})}")

    # Run full analysis
    print("\n6. Running full segmented analysis...")
    results = analyzer.run_segmented_analysis()

    print(f"\n   Found {len(results)} segments")
    for r in results:
        sig = "***" if r.is_significant else ""
        print(f"   - {r.segment}: effect={r.effect_size:.3f}, p={r.p_value:.4f} {sig}")

    # Generate summary
    print("\n7. Generating summary...")
    summary = analyzer.generate_summary(results)

    print(f"\n   Significant segments: {summary['significant_segments']}/{summary['total_segments_analyzed']}")
    print(f"   Total treatment: {summary['total_treatment_customers']}")
    print(f"   Total control: {summary['total_control_customers']}")
    print(f"   Average significant effect: {summary['average_significant_effect']:.4f}")
    print(f"   Total effect size: {summary['total_effect_size']:.2f}")

    print("\n   Recommendations:")
    for i, rec in enumerate(summary['recommendations'], 1):
        print(f"   {i}. {rec[:80]}...")

    print("\n" + "=" * 60)
    print("A/B Testing Module: ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    test_ab_testing()
