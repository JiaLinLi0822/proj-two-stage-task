# Planning in the tree search
This Repository is the codes for planning in the tree search task.

![image-20250714164257067](/Users/lijialin/Library/Application Support/typora-user-images/image-20250714164257067.png)

## Overview

There are two folders, for tree configuration 1 and configuration 2. In each folder, it provides a unified interface for fitting 14 different decision models to behavioral data, with automatic parameter bound management and easy model comparison capabilities.

## Files

- `model_configs.jl` - Model configuration definitions with parameter bounds
- `fitting.jl` - Main fitting script with flexible model selection
- `example.jl` - Usage examples and demonstrations
- `box.jl` - Parameter constraint management system

## Available Models

### Two-Stage Models

- **model1**: Primary two-stage independent paths model
- **model2**: Two-stage correlated paths model  
- **model3**: Two-stage independent paths with pruning
- **model4**: Two-stage correlated paths with pruning
- **model5**: Two-stage independent paths with single drift rate
- **model6**: Forward greedy search model
- **model14**: Two-stage with average-based first stage

### Backward Search Models

- **model7**: Backward search model
- **model8**: Backward search with shared parameters
- **model9**: Backward search with reset
- **model10**: Backward search with reset and shared parameters

### One-Stage Models

- **model11**: One-stage parallel integration with vigor
- **model12**: One-stage parallel integration with single vigor
- **model13**: One-stage parallel integration with vigor and rating noise

## Quick Start

### 1. Load the system

```julia
include("fitting.jl")
```

### 2. Show available models

```julia
show_models()
```

### 3. Fit a single model

```julia
results = run_model_fitting("model1")
```

### 4. Compare multiple models

```julia
model_names = ["model1", "model2", "model11"]
results, comparison = compare_models(model_names)
```

## Detailed Usage

### Fitting a Single Model

```julia
# Basic usage
results = run_model_fitting("model1")

# With custom settings
results = run_model_fitting("model11";
                           data_file="data/Tree2_v3.json",
                           output_file="custom_results.csv")
```

### Model Comparison

```julia
# Compare specific models
two_stage_models = ["model1", "model2", "model5"]
results, comparison = compare_models(two_stage_models)

# Compare vigor models
vigor_models = ["model11", "model12", "model13"]
results, comparison = compare_models(vigor_models)

# Compare backward search models
backward_models = ["model7", "model8", "model9", "model10"]
results, comparison = compare_models(backward_models)
```

### Inspecting Model Configuration

```julia
# Get model configuration
config = get_model_config("model1")

# Display parameter bounds
display(config.parameter_box)

# Get initial parameter values
x0, param_names = get_initial_params(config.parameter_box)

# Get BADS bounds
lb, ub, plb, pub, param_names = get_bads_bounds(config.parameter_box)
```

## Parameter Management

The system automatically manages parameter bounds for each model:

- **Log-scale parameters**: Drift rates (d, d1, d2, d0) and rating noise (rating_sd)
- **Linear-scale parameters**: Thresholds (θ, θ1, θ2), non-decision times (T1, T2), vigor parameters

### Example Parameter Bounds

For model1 (two-stage independent paths):

```julia
:d1 => (1e-10, 1e-3, :log)      # First stage drift rate
:d2 => (1e-10, 1e-3, :log)      # Second stage drift rate  
:θ1 => (1e-3, 2.0)              # First stage threshold
:θ2 => (1e-3, 2.0)              # Second stage threshold
:T1 => (10.0, 10000.0)          # First stage non-decision time
:T2 => (10.0, 10000.0)          # Second stage non-decision time
```

## Output Files

### Single Model Fitting

- CSV file with columns: `wid, [parameter_names], neglogl`
- Automatic timestamped filename: `results_model1_20231201_143022.csv`

### Model Comparison

- Individual CSV files for each model
- Summary comparison CSV with model rankings
- Automatic timestamped filenames

## Adding New Models

To add a new model:

1. **Define the model function** in `model.jl`
2. **Add configuration** to `MODEL_CONFIGS` in `model_configs.jl`:

```julia
"model15" => ModelConfig(
    model15,
    Box(:param1 => log_bounds(1e-10, 1e-3),
        :param2 => linear_bounds(0.1, 2.0)),
    "Description of model15"
)
```

## Testing

Run a quick test to verify everything works:

```julia
include("example_usage.jl")
quick_test("model1")
```

## Advanced Features

### Custom Parameter Bounds

Modify bounds in `model_configs.jl`:

```julia
# Example: Tighter bounds for model1
"model1_tight" => ModelConfig(
    model1,
    Box(:d1 => log_bounds(1e-8, 1e-4),     # Tighter drift bounds
        :d2 => log_bounds(1e-8, 1e-4), 
        :θ1 => linear_bounds(0.1, 1.0),    # Tighter threshold bounds
        :θ2 => linear_bounds(0.1, 1.0),
        :T1 => linear_bounds(100.0, 1000.0), # Tighter time bounds
        :T2 => linear_bounds(100.0, 1000.0)),
    "Model1 with tighter parameter bounds"
)
```

### Parallel Processing

The system automatically uses parallel processing:

```julia
# Modify number of workers in main_flexible.jl
addprocs(8)  # Use 8 workers instead of 4
```

### Custom Data Files

```julia
# Use different data file
results = run_model_fitting("model1"; data_file="data/custom_data.json")
```

## Troubleshooting

### Common Issues

1. **Model not found**: Check spelling and use `show_models()` to see available models
2. **Parameter errors**: Check model configuration in `model_configs.jl`
3. **Data loading errors**: Verify data file path and format
4. **Memory issues**: Reduce number of parallel workers

### Debug Mode

Add debug prints to track fitting progress:

```julia
# In objective_function, add:
println("Worker $(myid()): θ = $θ, negLL = $neg_ll")
```

## Performance Tips

1. **Use appropriate number of workers** based on your CPU cores
2. **Start with fewer subjects** for testing
3. **Use tighter parameter bounds** if you have prior knowledge
4. **Monitor memory usage** for large datasets

---

