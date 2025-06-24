# Artifact Repository for "Mind the Abstraction Gap: Bringing Equality Saturation to Real-World ML Compilers"

## Introduction
> In the Introduction, briefly explain the purpose of the artifact and how it supports the paper. We recommend listing all claims in the paper and stating whether or not each is supported. For supported claims, say how the artifact provides support. For unsupported claims, explain why they are omitted.

This artifact bundles **Constable**, an equality-saturation optimisation pass for XLA/StableHLO, together with the evaluation scripts needed to run the experiments from the paper.  It supports the following claims:

| Paper claim                                                                                       | Supported by the artifact? | How it is supported                                                                                    |
| ------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------ |
| **1. Constable delivers mean speed-up of ~3% (max 56.26 %) over stock XLA**                       | **Yes**                    | `experiments/baseline.sh` reproduces Fig. 7.                                                           |
| **2. Constable outperforms Enzyme-JAX by up to 54.82 %**                                          | **Yes**                    | `experiments/eqsat_vs_enzyme.sh` reproduces Fig. 8.                                                    |
| **3. Segmentation cuts exploration time while preserving most optimization potential** (Sec. 3.2, 5.5)  | **Yes**                    | `experiments/segmentation.sh` reproduces Fig. 10.                                                      |
| **4. First-order ILP cost model (fusion-aware) and zero-cost copies matter** (Sec. 3.3, 5.4)      | **Yes**                    | `experiments/cost_model_ablation.sh` toggles fusion bonus and copy cost and reproduces Fig. 9.         |
| **5. Constable generalises across ten diverse workloads (LLMs, ResNet, MD, KAN, …)**              | **Yes**                    | All workloads are evaluated in `experiments/baseline.sh`                                               |

All claims in the paper can be regenerated with the included scripts.
<!-- the only exception is the optional full-scale MaxText run on an A100, for which we supply a "small-batch" configuration that finishes in <10 min and preserves the optimisation behaviour. -->

## Hardware Dependencies
> In the Hardware Dependencies section, describe the hardware required to evaluate the artifact. If the artifact requires specific hardware (e.g., many cores, disk space, GPUs, specific processors), please provide instructions on how to gain access to the hardware. Keep in mind that reviewers must remain anonymous.

| Component | Minimum (quick sanity run)   | Recommended (full reproducibility)                         |
| --------- | ---------------------------- | ---------------------------------------------------------- |
| **CPU**   | 8 cores                      | 32-cores                                                   |
| **GPU**   | NVIDIA GPU (CUDA 12.6+)      | NVIDIA GPU with 24 GiB+ VRAM (Ampere or newer, CUDA 12.6+) |

The paper’s evaluation used A100 40 GB, V100 32 GB, RTX-3090 24 GB, and the corresponding host CPUs (Xeon E5-2603 v4 & Threadripper 3970X). We recommend a CPU with a higher core count as the setup process requires building LLVM, which takes a significant amount of CPU resources.

## Getting Started
> In the Getting Started Guide, give instructions for setup and basic testing. List any software requirements and/or passwords needed to access the artifact. The instructions should take roughly 30 minutes to complete. Reviewers will follow the guide during an initial kick-the-tires phase and report issues as they arise.

> The Getting Started Guide should be as simple as possible, and yet it should stress the key elements of your artifact. Anyone who has followed the Getting Started Guide should have no technical difficulties with the rest of your artifact.

Note that we assume that the target machine has a valid CUDA and GPU driver installation with CUDA 12.6+.

0. **Install NVIDIA container toolkit**

Note that CUDA 12.6 containers require an NVIDIA driver in the 525.85 (R525) branch or newer; NVIDIA recommends 560+. We will use toolkit version 1.17.8, which is the latest as of artifact submission.

```bash
# (Ubuntu example)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt-get install -y \
   nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
   libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

For more details and instructions for other Linux distros, see [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **Clone & pull the Docker image**

```bash
git clone https://github.com/smjleo/constable-oopsla25.git
cd docker
docker build -t constable:latest .
```
> [!NOTE]
> Building the Docker image will involve compiling LLVM, which can take a significant amount of time. On a Threadripper 3970X, the entire process took around 60 minutes; it may take longer on less powerful processors.

2. **Launch the container**

```bash
docker run --rm -it --gpus all constable:latest

# One specific GPU (example: GPU 1)
docker run --rm -it --gpus '"device=1"' constable:latest
```

Either `--gpus` or the environment variable `NVIDIA_VISIBLE_DEVICES` may be used. We recommend selecting a specific GPU on multi-GPU machines for consistent results.

3. **Verifying the installation**

```bash
JAX_PLATFORM=cpu EQSAT_PLATFORM=cpu python tests/llama.py # optimises llama on CPU; finishes in < 30 seconds
JAX_PLATFORM=gpu EQSAT_PLATFORM=gpu python tests/llama.py # optimises llama on GPU; finishes in < 30 seconds
```

Each script prints three numbers: XLA runtime, Enzyme's default optimization pipeline runtime (DefOpt) and Constable runtime.


## Step-by-Step Instructions
> In the Step by Step Instructions, explain how to reproduce any experiments or other activities that support the conclusions in your paper. Write this for readers who have a deep interest in your work and are studying it to improve it or compare against it. If your artifact runs for more than a few minutes, point this out, note how long it is expected to run (roughly) and explain how to run it on smaller inputs. Reviewers may choose to run on smaller inputs or larger inputs depending on available resources.

> Be sure to explain the expected outputs produced by the Step by Step Instructions. State where to find the outputs and how to interpret them relative to the paper. If there are any expected warnings or error messages, explain those as well. Ideally, artifacts should include sample outputs and logs for comparison.

Scripts for running experiments can be found in `Enzyme-JAX/experiment`. There are four scripts:
* `baseline.sh` measures the performance of Constable against XLA and Enzyme-JAX;
* `enzyme_vs_eqsat.sh` compares the fixpoint applier of Enzyme-JAX versus the equality saturation applier of Constable when constrained to the same set of rewrites;
* `cost_model.sh` compares the cost model used in equality saturation with ones that don't consider fusion costs or copy elision;
* `segmentation.sh` measures the runtime and optimisation time for various segment sizes.

In addition, we provide a script `Enzyme-JAX/graph_results.py` to produce graphs presented in the paper using the results generated by the experiment scripts.

We will describe the execution and graphing of each experiment in more detail.

### Baseline
#### Running
The baseline experiment measures the performance of Constable against XLA and Enzyme-JAX across all benchmark models.

```bash
cd ~/Enzyme-JAX
experiment/baseline.sh
```

This script will:
- Test all 10 models (`bert`, `gpt2`, `jaxmd`, `kan1`, `kan2`, `llama`, `maxtext`, `nasrnn`, `resnet`, `searchlesschess`)
- Run both CPU and GPU platforms
- Perform 9 repetitions
- Generate timestamped output text file
- Generate result `.csv` files for each benchmark and repeat

#### Customizing the Experiment

You can modify the experiment by editing the arrays at the top of `baseline.sh`:

```bash
platforms=("cpu" "gpu")
models=("bert" "gpt2" "jaxmd" "kan1" "kan2" "llama" "maxtext" "nasrnn" "resnet" "searchlesschess")
num_repeats=9
```

For example, to run a quick test on just GPU with fast models and fewer repetitions,
```bash
platforms=("gpu")
models=("llama" "nasrnn" "resnet")
num_repeats=3
```

#### Expected Runtime
**TODO** KAN1/KAN2, and also we should separate this by CPU/GPU

The experiment runtime varies significantly by model and platform:

**Fast models** (< 2 minutes per run):
- `bert`, `llama` `gpt2`, `nasrnn`, `resnet`, `searchlesschess`

**Slow models** (15+ minutes per run):
- `maxtext` - **particularly slow on CPU due to empirical cost model** (can take 40+ minutes per run)
- `jaxmd`

In total, the full experiment (all models, both platforms, 9 runs) can take up to ~14 hours. All repetitions for fast models only will take closer to 1-2 hours.

#### Fast vs Slow Experiment Variants
We provide the following variants that group fast and slow experiments together:
**Faster CPU and GPU models** (TODO minutes):
```bash
platforms=("cpu")
models=(TODO)
num_repeats=9
```
```bash
platforms=("gpu")
models=("bert" "gpt2" "kan1" "kan2" "llama" "nasrnn" "resnet")
num_repeats=9
```

**Slower CPU and GPU models** (TODO minutes):
```bash
platforms=("cpu")
models=(TODO)
num_repeats=9
```
```bash
platforms=("gpu")
models=("jaxmd" "maxtext" "searchlesschess")
num_repeats=9
```

#### Expected Outputs

The script generates several types of output:

1. Log file: `baseline_YYYY-MM-DD_HH:MM:SS.txt`
- Contains detailed execution logs
- Timing information for each experiment
- Error messages if any experiments fail

2. Performance CSVs (generated by `test_utils.py`): `results_{model}-{platform}_YYYY-MM-DD_HH:MM:SS_run{N}.csv`
- e.g. `results_bert-cpu_2025-06-24_10:46:56_run1.csv`
- Containing runtime measurements for JAX, Enzyme-JAX, and Constable

<!-- #### Troubleshooting -->
<!-- - If you encounter CUDA out-of-memory errors, try running CPU experiments first. Check for ghost GPU-hogging processes with `nvidia-smi` as well. -->
<!-- - MaxText on CPU is particularly resource-intensive; consider excluding it for initial testing -->

#### Graphing
**TODO**
* say how to run the graphing script given the result csvs
* Should mention that we only need to pass in one run, and it attempts to find all 9 runs based on the filename

### Constable vs Enzyme-JAX
**TODO** similar content as baseline

The process of running and graphing experiments is very similar to the baseline benchmarks. We highlight the main differences:

#### Running
```bash
cd ~/Enzyme-JAX
experiment/enzyme_vs_eqsat.sh
```

#### Expected Runtime
**TODO**

#### Fast vs Slow Variants
**TODO**

#### Expected output
**TODO**

#### Graphing
**TODO**

### Cost model ablation
**TODO** similar content as baseline

The process of running and graphing experiments is very similar to the baseline benchmarks. We highlight the main differences:

#### Running
```bash
cd ~/Enzyme-JAX
experiment/cost_model.sh
```

#### Expected Runtime
**TODO**

#### Fast vs Slow Variants
**TODO**

#### Expected output
**TODO**

#### Graphing
**TODO**

### Segmentation
#### Running
**TODO**
* Say something about segmentation size CSV being needed, and the ILP/saturation time limits are calculated using the number of segments + `compute_time_limits.py` automatically in the script
* Run experiment/segmentation.sh
* Again highlight which ones take long and how long + perhaps separate out the scripts.
* Again say which CSVs we expect (in particular we need the stats file too)

#### Graphing
**TODO**
* Say how to run the graphing script again. Running the script for segmentation is quite complicated because for each model, we require both the baseline CSV (to get the JAX numbers) and the segmentation CSV.
* Should also mention that we only need to pass in tau=5, and it attempts to find all segment sizes based on the filename

## Reusability Guide
### Using Constable with your own JAX program

Constable is implemented as an MLIR optimization pass called `"equality-saturation-pass"`. There are two main ways to use it:

#### Method 1: Direct usage

For applying Constable to a custom JAX function:

```python
import jax
import jax.numpy as jnp
from enzyme_ad.jax import enzyme_jax_ir, JaXPipeline

# Define your JAX function
def my_model(x, weights):
    h1 = jnp.dot(x, weights['w1'])
    h1_relu = jnp.maximum(h1, 0)
    output = jnp.dot(h1_relu, weights['w2'])
    return output

x = jax.random.normal(jax.random.PRNGKey(0), (32, 784))
weights = {
    'w1': jax.random.normal(jax.random.PRNGKey(1), (784, 128)),
    'w2': jax.random.normal(jax.random.PRNGKey(2), (128, 10))
}

constable_pipeline = JaXPipeline(
    "inline{default-pipeline=canonicalize max-iterations=4},"
    "equality-saturation-pass"
)

optimized_fn = jax.jit(
    enzyme_jax_ir(pipeline_options=constable_pipeline, inner_jit=False)(my_model)
)

result = optimized_fn(x, weights)
```

#### Method 2: Test harness

For systematic benchmarking and comparison, use the provided test framework:

```python
from absl.testing import absltest
from test_utils import *
import jax
import jax.numpy as jnp

def my_model(inputs, weights):
    x = jnp.dot(inputs, weights['linear'])
    x = jnp.tanh(x)
    return jnp.sum(x)

class MyModelTest(EnzymeJaxTest):
    def setUp(self):
        # Initialize your model inputs
        key = jax.random.PRNGKey(42)
        self.inputs = jax.random.normal(key, (128, 256))
        self.weights = {
            'linear': jax.random.normal(jax.random.split(key)[1], (256, 128))
        }

        # Required test harness parameters
        self.fn = my_model
        self.name = "my_model"
        self.count = 50  # Number of timing runs
        self.revprimal = False  # Set to True if testing gradients
        self.AllPipelines = pipelines()  # Use default pipelines
        self.AllBackends = CurBackends  # Use current backends

        # Input/output configuration for the harness
        self.ins = [self.inputs, self.weights]  # Function inputs
        self.dins = [self.inputs, self.weights]  # For gradient testing (if needed)
        self.douts = my_model(self.inputs, self.weights)  # Expected output
        self.tol = 1e-5  # Numerical tolerance

if __name__ == "__main__":
    from test_utils import fix_paths
    fix_paths()  # Setup CUDA paths
    absltest.main()
```


When running the test harness, you'll see output like:
```
my_model	JaX  	gpu	Primal	123.456 (min 120.1, max 130.2, mean 125.3 ± 2.1)
my_model	DefOpt	gpu	Primal	118.234 (min 115.5, max 122.1, mean 119.1 ± 1.8)
my_model	EqSat	gpu	Primal	98.765 (min 95.2, max 102.3, mean 99.8 ± 2.5)
```

This shows:
- model name, pipeline, platform, stage (Primal/Forward/PreRev/PostRev; only Primal is currently enabled)
- median runtime
- min/max/mean ± stdev

#### Environment Variables

You can control Constable's behavior using environment variables:

```bash
export EQSAT_PLATFORM=gpu  # or 'cpu'

export SEGMENTATION_THRESHOLD=200  # Segment size (default: 200)
export ILP_TIME_LIMIT=10           # ILP solver timeout (seconds, default: 100)
export SATURATION_TIME_LIMIT=10    # E-graph saturation timeout (seconds, default: 3600)

export FUSION_COSTS=true    # Enable fusion cost modeling (gpu only, default: true)
export ZERO_COSTS=true      # Assume zero cost for reshape/transpose ops (default: true)
```

In addition, the following environment variables control the behaviour of the test harness:

```bash
export EQSAT_ONLY=true    # Only run Constable, instead of with JAX and Enzyme-JAX (default: false)
export EXPERIMENT_NAME="bert-cpu_2025-06-24_10:46:56_run1"    # Dump results into results_{EXPERIMENT_NAME}.csv
```

### Extending Constable

#### File Structure and Components

Constable is implemented across C++ and Rust, and can be invoked in Python through the `enzyme_jax` package.

**TODO**
|Filename|Description|
|--------|-----------|
|`src/enzyme_ad/jax/Passes/EqualitySaturation.cpp`|Main C++ implementation of Constable. Responsible for calling Rust to build the e-graph, trigger optimization, and convert back to StableHLO. Also implements the empirical cost model and segmentation.|
|`src/enzyme_ad/jax/Passes/EqualitySaturation.h`|Functions exposed to Rust for calling the cost model.|
|TODO rust directory||

#### Adding New Rewrite Rules

**TODO**

#### Adding New StableHLO Operations

**TODO**

To support additional StableHLO operations:

1. **Add operation to encoding** (`src/enzyme_ad/jax/deps/src/input.rs`):
**TODO**

2. **Implement translation** from StableHLO to e-graph representation
**TODO**

3. **Add cost model** for the new operation
**TODO**

4. **Define rewrite rules** that can optimize the new operation
**TODO**
