# Large File Support with PySpark

## Overview

The A/B Testing Agent automatically detects file size and switches between two processing backends:

- **pandas** (default): For files ≤ 2MB - fast, in-memory processing
- **PySpark** (optional): For files > 2MB - distributed processing for big data

## How It Works

### Automatic Backend Selection

When you load a CSV file, the agent automatically:

1. Checks the file size
2. If the file is **larger than 2MB**, it switches to PySpark for distributed processing
3. If the file is **2MB or smaller**, it uses pandas for in-memory processing

This happens **automatically** - no user action required!

### Example Output

When you upload a large file, you'll see:

```
File size: 29.14 MB
[LARGE FILE DETECTED] Using PySpark for distributed processing (file size > 2MB)

Successfully loaded data from 'sample_ab_data_large.csv'
Shape: 500,000 rows, 5 columns
```

## Installation

### For Small Files Only (Default)

No additional setup needed! The default installation handles files up to 2MB:

```bash
uv pip install -r requirements.txt
```

### For Large Files (>2MB)

To enable PySpark for large file support:

1. **Uncomment PySpark in requirements.txt:**

   Edit `requirements.txt` and uncomment the last line:
   ```
   # Big Data Processing (optional - for files >2MB)
   pyspark>=3.5.0
   ```

2. **Install PySpark:**

   ```bash
   uv pip install pyspark>=3.5.0
   ```

3. **Done!** The agent will automatically use PySpark for files >2MB

## Testing Large File Support

### Using the Provided Test Dataset

A 29MB test dataset is included in `data/sample_ab_data_large.csv` with 500,000 rows.

To test:

1. Install PySpark (see above)
2. Run the Chainlit app:
   ```bash
   chainlit run app.py
   ```
3. Upload `data/sample_ab_data_large.csv`
4. The agent will automatically detect the large file and use PySpark

### Generating Custom Test Data

You can generate custom large datasets using:

```bash
python scripts/generate_large_sample_data.py
```

This creates a ~29MB CSV file with realistic A/B test data including:
- 500,000 rows
- Multiple customer segments (Premium, Standard, Basic, Trial, Enterprise)
- Pre/post effect columns for DiD analysis
- Realistic treatment effects varying by segment

## Performance Benefits

### pandas (Small Files ≤ 2MB)
- **Pros**: Fast for small datasets, simple, low overhead
- **Cons**: Limited by available RAM
- **Best for**: Quick analyses, files under 2MB

### PySpark (Large Files > 2MB)
- **Pros**: Handles massive datasets, distributed processing, optimized aggregations
- **Cons**: Higher startup overhead, requires more setup
- **Best for**: Large datasets, production workloads, files over 2MB

## Feature Parity

Both backends support the **exact same statistical analysis**:

✅ T-tests (Welch's t-test)
✅ Cohen's d effect size
✅ Statistical power analysis
✅ Two-proportion z-tests
✅ AA tests for balance checking
✅ Bootstrap balancing
✅ Bayesian A/B testing with Monte Carlo simulation
✅ Difference-in-Differences (DiD) causal inference
✅ Segmented analysis
✅ Auto-detection of columns

The only difference is **how** the data is processed internally - the results are identical.

## Customizing the Threshold

The 2MB threshold can be adjusted in `src/agent.py`:

```python
def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
    # ...
    self.FILE_SIZE_THRESHOLD_MB = 2  # Change this value
```

Common threshold values:
- `1` MB - More aggressive PySpark usage
- `5` MB - Use pandas for more files, PySpark only for very large files
- `10` MB - Maximize pandas usage, PySpark for huge files only

## Platform Compatibility

### pandas backend
✅ Windows
✅ Linux
✅ macOS

### PySpark backend
❌ Windows (limited support - see known issues)
✅ Linux
✅ macOS

**Note for Windows users:** PySpark has compatibility issues on Windows due to `socketserver.UnixStreamServer`. For large files on Windows, consider:
1. Using WSL (Windows Subsystem for Linux)
2. Processing files on a Linux/Mac machine
3. Increasing the threshold to avoid triggering PySpark

## Troubleshooting

### "PySpark not available" message

If you see this even after installing PySpark:

1. Verify installation:
   ```bash
   python -c "import pyspark; print(pyspark.__version__)"
   ```

2. Reinstall PySpark:
   ```bash
   uv pip install --force-reinstall pyspark>=3.5.0
   ```

3. Restart the Chainlit server

### PySpark errors on Windows

PySpark has limited Windows support. Options:
- Use WSL (recommended)
- Increase `FILE_SIZE_THRESHOLD_MB` to avoid PySpark
- Process large files on Linux/Mac

### Memory issues with large files

If you run out of memory even with PySpark:

1. Increase available RAM
2. Use Spark configuration to limit memory:
   ```python
   # In pyspark_analyzer.py, modify create_spark_session()
   .config("spark.driver.memory", "4g")  # Adjust as needed
   .config("spark.executor.memory", "4g")
   ```

## Examples

### Small File (pandas)
```
File: sample_ab_data.csv
File size: 0.03 MB
Using pandas for in-memory processing

Successfully loaded data from 'sample_ab_data.csv'
Shape: 1,500 rows × 5 columns
```

### Large File (PySpark)
```
File: sample_ab_data_large.csv
File size: 29.14 MB
[LARGE FILE DETECTED] Using PySpark for distributed processing (file size > 2MB)

Successfully loaded data from 'sample_ab_data_large.csv'
Shape: 500,000 rows × 5 columns
```

Both files will be analyzed with the same statistical rigor - the only difference is the processing backend!

## Additional Resources

- **PySpark Documentation**: https://spark.apache.org/docs/latest/api/python/
- **PySpark Analyzer Implementation**: [src/statistics/pyspark_analyzer.py](src/statistics/pyspark_analyzer.py)
- **Test Suite**: [tests/test_pyspark_analyzer.py](tests/test_pyspark_analyzer.py)
- **Test Results**: [TEST_RESULTS.md](TEST_RESULTS.md)
