"""
Generate a large sample A/B test dataset (~10MB) for testing PySpark analyzer

This creates a realistic dataset with:
- ~500,000 rows
- Multiple segments
- Pre/post effect columns for DiD analysis
- Treatment/control groups
"""

import pandas as pd
import numpy as np

def generate_large_ab_test_data(n_rows=500000, output_file='data/sample_ab_data_large.csv'):
    """Generate large A/B test dataset with realistic patterns"""

    np.random.seed(42)

    print(f"Generating {n_rows:,} rows of A/B test data...")

    # Generate data in chunks to avoid memory issues
    chunk_size = 50000
    chunks = []

    for i in range(0, n_rows, chunk_size):
        current_chunk_size = min(chunk_size, n_rows - i)

        # Customer segments with realistic distribution
        segments = np.random.choice(
            ['Premium', 'Standard', 'Basic', 'Trial', 'Enterprise'],
            size=current_chunk_size,
            p=[0.15, 0.35, 0.30, 0.15, 0.05]
        )

        # Treatment/control assignment (50/50 split)
        groups = np.random.choice(['treatment', 'control'], size=current_chunk_size)

        # Base revenue depends on segment
        segment_base_revenue = {
            'Premium': 150,
            'Standard': 80,
            'Basic': 40,
            'Trial': 10,
            'Enterprise': 300
        }

        base_revenue = np.array([segment_base_revenue[s] for s in segments])

        # Pre-effect values (baseline before experiment)
        pre_effect = base_revenue + np.random.normal(0, base_revenue * 0.3, current_chunk_size)
        pre_effect = np.maximum(pre_effect, 0)  # No negative revenue

        # Treatment effects vary by segment
        segment_treatment_lift = {
            'Premium': 0.08,      # 8% lift
            'Standard': 0.12,     # 12% lift
            'Basic': 0.15,        # 15% lift
            'Trial': 0.20,        # 20% lift
            'Enterprise': 0.05    # 5% lift (harder to move)
        }

        # Post-effect values
        post_effect = pre_effect.copy()

        # Apply treatment effect
        for idx, (segment, group) in enumerate(zip(segments, groups)):
            if group == 'treatment':
                # Treatment gets a lift
                lift = segment_treatment_lift[segment]
                post_effect[idx] = pre_effect[idx] * (1 + lift) + np.random.normal(0, 5)
            else:
                # Control stays similar with natural variation
                post_effect[idx] = pre_effect[idx] + np.random.normal(0, 5)

        post_effect = np.maximum(post_effect, 0)  # No negative revenue

        # Create chunk dataframe
        chunk_df = pd.DataFrame({
            'customer_id': range(i, i + current_chunk_size),
            'segment': segments,
            'group': groups,
            'pre_effect': pre_effect,
            'post_effect': post_effect
        })

        chunks.append(chunk_df)

        if (i + chunk_size) % 100000 == 0:
            print(f"  Generated {i + chunk_size:,} rows...")

    # Combine all chunks
    df = pd.concat(chunks, ignore_index=True)

    # Shuffle rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_file, index=False)

    # Print statistics
    file_size_mb = len(df.to_csv(index=False).encode('utf-8')) / (1024 * 1024)

    print(f"\n[SUCCESS] Generated {len(df):,} rows")
    print(f"[SUCCESS] File size: ~{file_size_mb:.1f} MB")
    print(f"[SUCCESS] Saved to: {output_file}")
    print(f"\nDataset statistics:")
    print(f"  Treatment: {(df['group'] == 'treatment').sum():,} ({(df['group'] == 'treatment').mean()*100:.1f}%)")
    print(f"  Control: {(df['group'] == 'control').sum():,} ({(df['group'] == 'control').mean()*100:.1f}%)")
    print(f"\nSegment distribution:")
    print(df['segment'].value_counts().sort_index())
    print(f"\nRevenue statistics:")
    print(f"  Pre-effect mean: ${df['pre_effect'].mean():.2f}")
    print(f"  Post-effect mean: ${df['post_effect'].mean():.2f}")
    print(f"  Treatment post mean: ${df[df['group']=='treatment']['post_effect'].mean():.2f}")
    print(f"  Control post mean: ${df[df['group']=='control']['post_effect'].mean():.2f}")

    return df

if __name__ == '__main__':
    df = generate_large_ab_test_data(n_rows=500000)
