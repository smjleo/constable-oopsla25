# Artifact Repository for "Mind the Abstraction Gap: Bringing Equality Saturation to Real-World ML Compilers"

## Introduction
<!-- > In the Introduction, briefly explain the purpose of the artifact and how it supports the paper. We recommend listing all claims in the paper and stating whether or not each is supported. For supported claims, say how the artifact provides support. For unsupported claims, explain why they are omitted. -->

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
<!-- > In the Hardware Dependencies section, describe the hardware required to evaluate the artifact. If the artifact requires specific hardware (e.g., many cores, disk space, GPUs, specific processors), please provide instructions on how to gain access to the hardware. Keep in mind that reviewers must remain anonymous. -->

| Component | Minimum                      | Recommended                                                |
| --------- | ---------------------------- | ---------------------------------------------------------- |
| **CPU**   | 8 cores                      | 16 cores                                                   |
| **GPU**   | NVIDIA GPU (CUDA 12.6+)      | NVIDIA GPU with 24 GiB+ VRAM (Ampere or newer, CUDA 12.6+) |
| **Storage** | 45GB | 60GB |

The paper’s evaluation used A100 40 GB, V100 32 GB, RTX-3090 24 GB, and the corresponding host CPUs (Xeon E5-2603 v4 & Threadripper 3970X).
If you choose to build the Docker image yourself, we recommend a CPU with a higher core count as the setup process requires compiling LLVM from source, which takes a significant amount of CPU resources.

> [!NOTE]
> We recommend avoiding the use of virtualized CPU cores, especially when benchmarking (use bare metal instead). Constable uses ortools / SCP ILP solver which is quite demanding on CPUs. We find that virtual CPUs often underperform bare metal in this setting, which can lead to timeouts in the ILP solving.

The experiments were run on Ubuntu 24.04 and Debian 12.

## Getting Started
<!-- > In the Getting Started Guide, give instructions for setup and basic testing. List any software requirements and/or passwords needed to access the artifact. The instructions should take roughly 30 minutes to complete. Reviewers will follow the guide during an initial kick-the-tires phase and report issues as they arise. -->

<!-- > The Getting Started Guide should be as simple as possible, and yet it should stress the key elements of your artifact. Anyone who has followed the Getting Started Guide should have no technical difficulties with the rest of your artifact. -->

Note that we assume that the target machine has a valid CUDA and GPU driver installation with CUDA 12.6+.

0. (If not already set up) **Install NVIDIA container toolkit**

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
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For more details and instructions for other Linux distros, see [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

1. **Launch Docker container**
> [!NOTE]
> If you run into permission issues, run the docker commands with `sudo` or add the current user to the [`docker` group](https://docs.docker.com/engine/install/linux-postinstall/).

> [!NOTE]
> We recommend selecting a specific GPU on multi-GPU machines for consistent results.

### Pre-built image
We recommend you use the pre-built Docker image `smjleo/constable-oopsla25`:

```bash
docker run --rm -it --gpus all --entrypoint /bin/bash smjleo/constable-oopsla25:latest

# One specific GPU (example: GPU 1)
docker run --rm -it --gpus '"device=1"' --entrypoint /bin/bash smjleo/constable-oopsla25:latest
```

### Build image
> [!WARNING]
> Building the Docker image will involve compiling LLVM, which can take a significant amount of time. On a Threadripper 3970X, the entire process took around 60 minutes; it may take longer on less powerful processors.

To build the image yourself, first clone and pull the Docker image:
```bash
git clone https://github.com/smjleo/constable-oopsla25.git
cd constable-oopsla25/docker
docker build -t constable:latest .
```

and launch the container.

```bash
docker run --rm -it --gpus all --entrypoint /bin/bash constable:latest

# One specific GPU (example: GPU 1)
docker run --rm -it --gpus '"device=1"' --entrypoint /bin/bash constable:latest
```

2. **Verify the installation**

```bash
JAX_PLATFORMS=cpu EQSAT_PLATFORM=cpu python test/llama.py # optimises llama on CPU; finishes in < 30 seconds
JAX_PLATFORMS=gpu EQSAT_PLATFORM=gpu python test/llama.py # optimises llama on GPU; finishes in < 30 seconds
```

Each script prints three numbers: XLA runtime, Enzyme's default optimization pipeline runtime (DefOpt), and Constable runtime.

## Step-by-Step Instructions
<!-- > In the Step by Step Instructions, explain how to reproduce any experiments or other activities that support the conclusions in your paper. Write this for readers who have a deep interest in your work and are studying it to improve it or compare against it. If your artifact runs for more than a few minutes, point this out, note how long it is expected to run (roughly) and explain how to run it on smaller inputs. Reviewers may choose to run on smaller inputs or larger inputs depending on available resources. -->

<!-- > Be sure to explain the expected outputs produced by the Step by Step Instructions. State where to find the outputs and how to interpret them relative to the paper. If there are any expected warnings or error messages, explain those as well. Ideally, artifacts should include sample outputs and logs for comparison. -->

Scripts for running experiments can be found in `Enzyme-JAX/experiment`. There are four scripts:
* `baseline.sh` measures the performance of Constable against XLA and Enzyme-JAX;
* `enzyme_vs_eqsat.sh` compares the fixpoint applier of Enzyme-JAX versus the equality saturation applier of Constable when constrained to the same set of rewrites;
* `cost_model.sh` compares Constable's cost model against alternatives that don't consider fusion costs or copy elision;
* `segmentation.sh` measures the runtime and optimisation time when performing equality saturation on various segment sizes.

In addition, we provide a script `Enzyme-JAX/graph_results.py` to produce graphs presented in the paper using the results generated by the experiment scripts.

### Performance stability guidance

In order to reduce variance in experimentation, we recommend disabling turbo boost, hyperthreading, and other CPU features that may lead to nondeterminism during benchmarking. Here are our recommended steps for Intel and AMD systems.

#### Intel Systems

First, we recommend disabling hyperthreading. We used the below snippet

```bash
#!/usr/bin/env bash
declare -A seen

# Read "cpu,core,socket,node" from lscpu; disable secondary threads
while IFS=',' read -r cpu core socket node; do
    key="${socket}_${core}"
    if [[ ${seen[$key]} ]]; then
        echo "Disabling hyperthreading logical core $cpu (Socket $socket, Core $core)"
        echo 0 | sudo tee /sys/devices/system/cpu/cpu${cpu}/online >/dev/null
    else
        seen[$key]=1
    fi
done < <(lscpu -p=CPU,Core,Socket,Node | grep -v '^#')
```

This ensures only one logical thread per physical core remains online. Running `lscpu | grep Off-line` should show thread siblings are offline, if this step worked successfully. Next, we recommend disabling turbo boost.

```bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
```

Note that some machines will show that the `/sys/devices/system/cpu/intel_pstate/` path doesn't exist. In this case, you might need to use acpi_cpufreq like below

```bash
sudo cpupower frequency-set --governor performance
sudo cpupower set -b 0
```

The former method turns off all turbo states, and the latter locks the CPU to the max non-turbo frequency.

#### AMD Systems

As with the Intel CPUs, we recommend disabling hyperthreading. Depending on kernel/Linux/driver versions, there are numerous ways to do this. We include some options below; please not that this is not exhaustive. You might need to disable SMT at the BIOS level in some cases.
```bash
echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null
```

We can disable CPU boost with amd-pstate:
```bash
echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
```
or with `acpi_cpufreq` if cpupower is installed.

```bash
sudo cpupower frequency-set --governor performance
sudo cpupower frequency-set --max 3500000 # this is 3.5 Ghz
```
where you input your CPU’s base clock in place of the above example frequency. 

#### Pinning processes

After disabling hyperthreading and frequency boost, when running the Docker image, preface your command with `taskset` to bind the process to a fixed core set, like below

```bash
# if we want to bind to cores 1–15
taskset -c 1-15 docker 
```

Adjust the core list based on your hardware. We strongly recommend excluding core 0 from your core set, since the OS will forcibly de-schedule off core 0 occasionally. One should also aim to pick a core set that lies entirely in one NUMA node, if possbile. NUMA nodes can be checked through the below command, courtesy of [this](https://www.baeldung.com/linux/lscpu-command-tutorial) lscpu tutorial.

```bash
lscpu -e | awk 'NR>1 {print "CPU "$1" is in NUMA node "$2}'
```
---

We now describe the execution and graphing of each experiment in more detail.

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
- Generate a timestamped log text file
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
| Model            | CPU Runtime (min) | GPU Runtime (min) |
|------------------|-------------------|-------------------|
| bert            | 2               | 4               |
| gpt2            | 2               | 2               |
| jaxmd           | 1               | 2               |
| kan1            | <1              | 2               |
| kan2            | <1              | 2               |
| llama           | <1              | <1              |
| maxtext         | 20              | <1              |
| nasrnn          | <1              | <1              |
| resnet          | <1              | <1              |
| searchlesschess | <1              | <1              |

In total, the full experiment (all models, both platforms, 9 runs) can take up to ~14 hours. All repetitions for fast models only will take closer to 1-2 hours.

> [!NOTE]
> The default exploration and extraction (ILP) time limits are 10 seconds.
> However, due to a bug in the OR-Tools solver, when `ILP_TIME_LIMIT` is small, `kan1` and `kan2` nondeterministically fail during extraction. To prevent this, we set `ILP_TIME_LIMIT=30` for these KAN networks during the baseline, cost model, and enzyme vs. eqsat ablations. We will make this clear in the revision of the manuscript.

#### Fast vs Slow Experiment Variants
We provide the following variants that group fast (<=1 min per repetition) and slow (>1 min per repetition) experiments together:
##### Fast
```bash
platforms=("cpu")
models=("jaxmd" "kan1" "kan2" "llama" "nasrnn" "resnet" "searchlesschess")
num_repeats=9
```
```bash
platforms=("gpu")
models=("llama" "maxtext" "nasrnn" "resnet" "searchlesschess")
num_repeats=9
```
##### Slow
```bash
platforms=("cpu")
models=("bert" "gpt2" "maxtext")
num_repeats=9
```
```bash
platforms=("gpu")
models=("bert" "gpt2" "jaxmd" "kan1" "kan2")
num_repeats=9
```

#### Expected output

The script generates several types of output:

1. Log file: `baseline_YYYY-MM-DD_HH:MM:SS.txt`
- Contains detailed execution logs
- Timing information for each experiment
  - Error messages if any experiments fail

2. Performance CSVs (generated by `test_utils.py`): `results_{model}-{platform}_YYYY-MM-DD_HH:MM:SS_run{N}.csv`
- e.g. `results_bert-cpu_2025-06-24_10:46:56_run1.csv`
- Containing runtime measurements for JAX, Enzyme-JAX, and Constable

3. Stats CSV: `stats_baseline_YYYY-MM-DD_HH:MM:SS.csv`
- Contains optimisation time, number of segments and number of while loops per experiment.

<!-- #### Troubleshooting -->
<!-- - If you encounter CUDA out-of-memory errors, try running CPU experiments first. Check for ghost GPU-hogging processes with `nvidia-smi` as well. -->
<!-- - MaxText on CPU is particularly resource-intensive; consider excluding it for initial testing -->

#### Graphing
We can invoke the graphing script in the `compare` mode, with the result CSVs:
```bash
./graph_results.py compare \
-p Xeon \
BERT results_bert-cpu_2025-06-24_10:46:56_run1.csv \
GPT-2 results_gpt2-cpu_2025-06-24_10:46:56_run1.csv \
JAX-MD results_jaxmd-cpu_2025-06-24_10:46:56_run1.csv \
...
-p A100 \
BERT results_bert-gpu_2025-06-24_10:46:56_run1.csv \
GPT-2 results_gpt2-gpu_2025-06-24_10:46:56_run1.csv \
JAX-MD results_jaxmd-gpu_2025-06-24_10:46:56_run1.csv \
...
```

The `-p <title>` flag creates a new plot. The subsequent arguments until the next `-p` provides the data for that plot.
For each model, we need to provide a pair `<model-name> <result-csv-run1>`. Crucially, we only pass in the CSV for the first run (out of 9 for the baseline; filename ends in `_run1.csv`). The graphing script parses the filename, and expects to find the other runs in the same directory.

> [!NOTE]
> The graphing script outputs (on standard output) the files it is using for each model.
> It would be a good idea to observe the output to ensure that all relevant files have been loaded.

### Constable vs Enzyme-JAX
The process of running and graphing experiments is very similar to the baseline benchmarks. We highlight the main differences:

#### Running
We run the `enzyme_vs_eqsat.sh` script.

```bash
cd ~/Enzyme-JAX
experiment/enzyme_vs_eqsat.sh
```

#### Expected Runtime
See the above table with runtime expectations for the baseline benchmarks. This ablation has similar runtime characteristics.

#### Expected output
Similarly to the baseline output, we expect:
1. A log file (`enzyme_vs_eqsat_YYYY-MM-DD_HH:MM:SS.txt`)
2. Performance CSVs (`results_{model}_enzyme-ablation-{platform}_YYYY-MM-DD_HH:MM:SS_run{N}.csv`).
3. A stats CSV (`stats_enzyme_vs_eqsat_YYYY-MM-DD_HH:MM:SS.csv`)

#### Graphing
We can use the graphing script in the same way as baseline benchmarks (making sure to pass in `enzyme_ablation` result CSVs).
```bash
./graph_results.py compare \
-p Xeon \
BERT results_bert_enzyme-ablation-cpu_2025-03-25_22:45:26_run1.csv \
GPT-2 results_gpt2_enzyme-ablation-cpu_2025-03-25_22:45:26_run1.csv \
...
-p A100 \
BERT results_bert_enzyme-ablation-gpu_2025-03-25_22:45:26_run1.csv \
GPT-2 results_gpt2_enzyme-ablation-gpu_2025-03-25_22:45:26_run1.csv \
...
```

### Cost model ablation
The process of running and graphing experiments is very similar to the baseline benchmarks. We highlight the main differences:

#### Running
```bash
cd ~/Enzyme-JAX
experiment/cost_model.sh
```

#### Expected Runtime
As before, the optimization runtime is similar to the baseline. See the above table.

#### Expected output
Similarly to the baseline output, we expect:
1. A log file (`cost_model_YYYY-MM-DD_HH:MM:SS.txt`)
2. Performance CSVs (`results_{model}_cost-model_{config_name}-{platform}_YYYY-MM-DD_HH:MM:SS_run{N}.csv`).
- `config_name` is one of `baseline`, `no-fusion`, or `no-zero` (for the default cost model, cost model without fusion costs, and cost model without copy ellision assumptions, respectively).
3. A stats CSV (`stats_cost_model_YYYY-MM-DD_HH:MM:SS.csv`)

#### Graphing
The invocation of the graph script is analogous to baseline benchmarks, but we instead use the `cost_model` mode.
```bash
./graph_results.py cost_model \
-p Xeon \
BERT results_bert_cost-model_baseline-cpu_2025-03-26_04:04:46_run1.csv \
GPT-2 results_gpt2_cost-model_baseline-cpu_2025-03-26_04:04:46_run1.csv \
...
-p A100 \
BERT results_bert_cost-model_baseline-gpu_2025-03-26_04:04:46_run1.csv \
GPT-2 results_gpt2_cost-model_baseline-gpu_2025-03-26_04:04:46_run1.csv \
...
```

### Segmentation
#### Running
We require, as an input to the segmentation experiment, a statistics CSV `stats_segmentation.csv` that counts the number of segments produced under different values of tau, for a given model. Please set the `segmentation_size_csv` variable within `segmentation.sh` to the name/path towards this `.csv` file.

For convenience, we provide `stats_segmentation.csv` in the Enzyme-JAX repository that provides this information.
To generate a new stats file (it should be identical to `stats_segmentation.csv`), simply run the `generate_stats_for_segmentation.sh` script. It runs each model through Constable with trivial time limits and rewrites in order to record the number of segments for each tau value. For context, we invoke a Python script `compute_time_limits.py` in `segmentation.sh` that uses this file to calculate the ILP/saturation time limits. See `compute_time_limits.py`, which is called in `segmentation.sh`, for more details on this process.

Once you have `stats_segmentation.csv`:
```bash
cd Enzyme-JAX
experiment/segmentation.sh
```

#### Expected outputs
This script generates several files:
1. A log file (`stats_segmentation_YYYY-MM-DD_HH.txt`)
2. Performance CSVs (`results_{model}_tau={segmentation_threshold}-{platform}_YYYY-MM-DD_HH:MM:SS.csv`)
* For $\tau = 5, 10, 25, 100, 500, 1000, 5000$
3. A stats CSV (`stats_segmentation_YYYY-MM-DD_HH.csv`)

#### Graphing
Graphing segmentation is more involved due to the amount of data involved. An example usage is as follows:
```bash
./graph_results.py segmentation \
-p Xeon \
BERT results_bert-cpu_2025-06-24_10:46:56_run1.csv results_bert_tau=5-cpu_2025-03-24_09:48:27.csv \
GPT-2 results_gpt2-cpu_2025-06-24_10:46:56_run1.csv results_gpt2_tau=5-cpu_2025-03-24_09:48:27.csv \
...
-stats stats_segmentation_2025-03-24_09:48:27.csv \
-p A100 \
BERT results_bert-gpu_2025-06-24_10:46:56_run1.csv results_bert_tau=5-gpu_2025-03-24_09:48:27.csv \
GPT-2 results_gpt2-gpu_2025-06-24_10:46:56_run1.csv results_gpt2_tau=5-gpu_2025-03-24_09:48:27.csv \
...
-stats stats_segmentation_2025-03-24_09:48:27.csv \
...
```

As before, we use the `-p` flag to create new plots. However, for each model, we are now required to pass in *two* files: the baseline performance CSV and the segmentation results CSV (for $\tau=5$). The baseline CSV is used to get the runtime for JAX for a given model, so that relative speedup can be measured (the segmentation experiment script does not run JAX measurements to conserve time). Similarly to the above experiments involving multiple runs, the graphing script expects to find other segment size CSVs in the same directory as the $\tau=5$ CSV.

> [!NOTE]
> The graphing script outputs (on standard output) the filenames it found for different segment sizes.
> It would be a good idea to observe the output to ensure that all relevant files have been loaded.

In addition, we pass in the `stats` file generated by the experiment via the `-stats` flag for each plot. This file contains the total optimisation time for each experiment, and is needed to plot the optimisation plot lines (blue lines in the graph).

#### Expected runtime
The segmentation ablation, unlike the other ablations, has significantly longer runtime cost compared to the baseline. We provide the expected total runtime across all segments for each model below:

| Model            | CPU Runtime (min) | GPU Runtime (min) |
|------------------|-------------------|---------------------|
| **BERT**         | 3                 | 18              |
| **GPT-2**        | 12                | 28              |
| **JAX-MD**       | 5                 | 5               |
| **KAN1**         | 1                 | 23              |
| **KAN2**         | 1                 | 20              |
| **Llama**        | 3                 | 4               |
| **MaxText**      | 135               | 2               |
| **NasRNN**       | 1                 | 1               |
| **ResNet**       | 2                 | 1               |
| **SearchlessChess** | 3              | 13              |

### Sample logs
We provide sample logs to show rough expected output formats.

#### Stats CSVs
```csv
experiment_name,eqsat_time,segments,while_loops
bert_cost-model_baseline-cpu_2025-03-25_12:33:11_run1,36.837,7,0
bert_cost-model_baseline-gpu_2025-03-25_12:33:11_run1,116.87,7,0
gemma_cost-model_baseline-cpu_2025-03-25_12:33:11_run1,34.534,3,0
gemma_cost-model_baseline-gpu_2025-03-25_12:33:11_run1,4.416,3,0
gpt2_cost-model_baseline-cpu_2025-03-25_12:33:11_run1,37.318,8,0
gpt2_cost-model_baseline-gpu_2025-03-25_12:33:11_run1,58.82,8,0
...
```

#### Result CSVs
```csv
pipeline,stage,runtime_ms
JaX  ,Primal,28.84198701940477
JaX  ,Primal,27.827179990708828
...
DefOpt,Primal,25.62545903492719
DefOpt,Primal,26.201438042335212
...
EqSat,Primal,25.178318028338253
EqSat,Primal,24.80236196424812
```

#### Log files
```
Cost model ablation
--------------------------
Running bert_cost-model_baseline-cpu_2025-03-25_12:33:11_run1 (repeat 1/12)...
Some weights of FlaxBertModel were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: {('pooler', 'dense', 'kernel'), ('pooler', 'dense', 'bias')}
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Running tests under Python 3.11.8: /home/pokemon/.conda/envs/pokemon/bin/python
[ RUN      ] BertTransformerTest.test
The current directory is /home/pokemon/Enzyme-JAX
Number of canonicalized 1
Run one
Done one
Number of nodes added: 0
Run one
Done one
Number of nodes added: 0
Runner complete!
  Nodes: 1581
  Classes: 1154
  Stopped: Saturated
  Time taken: 4.189377411s
  Number of iterations: 15
  Average nodes per class: 1.3700173
  Number of edges: 3054
  Number of programs: 318.3365
prepped ilp data
running with fusion costs
Set number of threads to 8
Set time limit to 10 seconds
Number of variables = 2735
Add eclass constraints
Objective value = 1489.0
Problem solved in 397.000000 milliseconds
Problem solved in 0 iterations
Problem solved in 1 branch-and-bound nodes
```

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

| Filename | Description |
|----------|-------------|
| `src/enzyme_ad/jax/Passes/EqualitySaturation.cpp` | Main C++ implementation of Constable. Responsible for calling Rust to build the e-graph, trigger optimisation, and convert back to StableHLO. Also implements the empirical cost model and segmentation.|
| `src/enzyme_ad/jax/Passes/EqualitySaturation.h` | Functions exposed to Rust for calling the cost model.|
| `src/enzyme_ad/jax/deps/tensat/src/ffi_utils.rs` | Helper functions to facilitate FFI calls between C++ and Rust |
| `src/enzyme_ad/jax/deps/tensat/src/input.rs` | Exposes functions for the construction of the e-graph from the C++ side (`CppGraphConverter::new_{node}_op`). Also exposes functions that the C++ side can call to trigger e-graph optimisation (`CppGraphConvert::optimize`).
| `src/enzyme_ad/jax/deps/tensat/src/model.rs` | Defines the e-graph language (i.e. the types of e-nodes), and the e-graph analysis. |
| `src/enzyme_ad/jax/deps/tensat/src/optimize.rs` | Exposes functions that call the cost model and triggers extraction. |
| `src/enzyme_ad/jax/deps/tensat/src/rewrites.rs` | Defines functions that parses the rewrite file and applies it to the e-graph. Also defines functions that load Enzyme-JAX rewrites. |
| `src/enzyme_ad/jax/deps/tensat/converted.txt` | Simple single-pattern rewrites |
| `src/enzyme_ad/jax/deps/tensat/converted_multi.txt` | Multi-pattern rewrites |

#### Adding New Rewrite Rules
There are three ways of adding new rewrite rules:
* If adding simple, syntax-directed rules, we can add them in `src/enzyme_ad/jax/deps/tensat/converted.txt` in the format `{pattern_lhs}=>{pattern_rhs}`, or `{pattern_lhs}<=>{pattern_rhs}` for bidirectional rules.
* If adding syntax-directed *multi-pattern* rules, we can add them in `src/enzyme_ad/jax/deps/tensat/converted_multi.txt`. The format is the same as `converted_txt`, except the rules come in pairs. See [Tensat](https://github.com/uwplse/tensat) for a more detailed description of multi-pattern rewrites.
* If adding rewrites from Enzyme, we can add the corresponding name in the `MlirRewrites` enum in `src/enzyme_ad/jax/deps/tensat/src/rewrites.rs`, and add the corresponding *abstract pattern* (pattern to be matched without any conditionals or attributes) in `MlirRewrites::to_ast`.

#### Adding New StableHLO Operations
To support additional StableHLO operations:

1. Add operation to model:
* In `model.rs`, add the operation to the `Mdl` enum, and change `clone_to_mapping`.
* If the node can be fused, change `fus_i` in `prep_ilp_data` within `optimize.rs`.

2. Implement translation from StableHLO to e-graph representation
* In `input.rs`, add the operation to the `Ops` enum, and create the corresponding `new_{node}_op_` function. Expose its signature in the `ffi` module (under `extern "Rust"`). Change `convert_to_node`.
* Add the node to `recexpr_to_node` in `ffi_utils.rs`
* In `EqualitySaturation.cpp`, change the `dfs` function to convert the new node into the e-graph representation, using the constructs defined above. Also change `reconstructStablehlo` and `createStableHloOp` which converts the e-graph node back to StableHLO. Change `isBlackboxed` to ensure the node is not blackboxed (some assertions will fail during execution otherwise). If the node should be a zero-cost op, add it to `zeroCostOps`.

3. Define rewrite rules that can optimize the new operation: see the above section.

