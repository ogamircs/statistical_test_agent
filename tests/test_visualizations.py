"""Test the visualization module"""

from ab_testing import ABTestAnalyzer, ABTestVisualizer

def test_visualizations():
    print("Testing A/B Test Visualizations...")
    print("=" * 60)

    # Initialize
    analyzer = ABTestAnalyzer()
    visualizer = ABTestVisualizer()

    # Load sample data
    print("\n1. Loading data...")
    analyzer.load_data("sample_ab_data.csv")

    # Configure
    print("2. Configuring columns...")
    analyzer.set_column_mapping({
        "customer_id": "customer_id",
        "group": "experiment_group",
        "effect_value": "effect_value",
        "segment": "customer_segment"
    })
    analyzer.set_group_labels("treatment", "control")

    # Run analysis
    print("3. Running analysis...")
    results = analyzer.run_segmented_analysis()
    summary = analyzer.generate_summary(results)

    print(f"   Found {len(results)} segments")
    print(f"   Significant: {summary['significant_segments']}")

    # Generate charts
    print("\n4. Generating charts...")
    charts = visualizer.create_all_charts(
        results, summary,
        df=analyzer.df,
        group_col="experiment_group",
        segment_col="customer_segment"
    )

    print(f"   Generated {len(charts)} charts:")
    for name, fig in charts.items():
        print(f"   - {name}: {type(fig).__name__}")

    # Test individual charts
    print("\n5. Testing individual chart functions...")

    fig1 = visualizer.plot_treatment_vs_control(results)
    print(f"   - Treatment vs Control: OK ({len(fig1.data)} traces)")

    fig2 = visualizer.plot_effect_sizes(results)
    print(f"   - Effect Sizes: OK ({len(fig2.data)} traces)")

    fig3 = visualizer.plot_p_values(results)
    print(f"   - P-Values: OK ({len(fig3.data)} traces)")

    fig4 = visualizer.plot_sample_sizes(results)
    print(f"   - Sample Sizes: OK ({len(fig4.data)} traces)")

    fig5 = visualizer.plot_power_analysis(results)
    print(f"   - Power Analysis: OK ({len(fig5.data)} traces)")

    fig6 = visualizer.plot_cohens_d(results)
    print(f"   - Cohen's d: OK ({len(fig6.data)} traces)")

    fig7 = visualizer.plot_summary_dashboard(results, summary)
    print(f"   - Dashboard: OK ({len(fig7.data)} traces)")

    fig8 = visualizer.plot_effect_waterfall(results)
    print(f"   - Waterfall: OK ({len(fig8.data)} traces)")

    # Save a sample chart to HTML for verification
    print("\n6. Saving sample chart to HTML...")
    fig7.write_html("test_dashboard.html")
    print("   Saved dashboard to test_dashboard.html")

    print("\n" + "=" * 60)
    print("Visualization Tests: ALL PASSED!")
    return True

if __name__ == "__main__":
    test_visualizations()
