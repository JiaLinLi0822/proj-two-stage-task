# setup_julia_env.jl - Julia environment configuration script

using Pkg

println("Starting Julia environment setup...")

# Check current Julia version
println("Julia version: $(VERSION)")

# Activate current project environment
Pkg.activate(".")

# Required packages list
required_packages = [
    "Distributed",
    "DataFrames", 
    "CSV",
    "JSON",
    "Dates",
    "Statistics",
    "Random",
    "LinearAlgebra"
]

println("Installing required packages...")

# Install packages
for pkg in required_packages
    try
        println("Checking package: $pkg")
        Pkg.add(pkg)
    catch e
        println("Failed to install $pkg: $e")
    end
end

# Precompile all packages
println("Precompiling packages...")
try
    Pkg.precompile()
    println("✓ Precompilation completed")
catch e
    println("⚠ Precompilation failed: $e")
end

# Test key package imports
println("Testing package imports...")
test_packages = [
    "Distributed",
    "DataFrames", 
    "CSV",
    "JSON"
]

for pkg in test_packages
    try
        eval(Meta.parse("using $pkg"))
        println("✓ $pkg")
    catch e
        println("✗ $pkg import failed: $e")
    end
end

# Check project files
project_files = [
    "fitting.jl",
    "model_configs.jl", 
    "data.jl",
    "ibs.jl",
    "bads.jl"
]

println("Checking project files...")
for file in project_files
    if isfile(file)
        println("✓ $file")
    else
        println("⚠ $file not found")
    end
end

# Check data file
if isfile("data/Tree1_v3.json")
    println("✓ Data file exists")
else
    println("⚠ Data file not found: data/Tree1_v3.json")
end

# Create necessary directories
dirs = ["results/Tree1", "logs"]
for dir in dirs
    if !isdir(dir)
        mkpath(dir)
        println("Created directory: $dir")
    else
        println("✓ Directory exists: $dir")
    end
end

println("Julia environment setup completed!")
