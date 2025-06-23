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

All claims in the paper can be regenerated with the included scripts; the only exception is the optional full-scale MaxText run on an A100, for which we supply a "small-batch" configuration that finishes in <10 min and preserves the optimisation behaviour.

## Hardware Dependencies
> In the Hardware Dependencies section, describe the hardware required to evaluate the artifact. If the artifact requires specific hardware (e.g., many cores, disk space, GPUs, specific processors), please provide instructions on how to gain access to the hardware. Keep in mind that reviewers must remain anonymous.

| Component | Minimum (quick sanity run)   | Recommended (full reproducibility)                         |
| --------- | ---------------------------- | ---------------------------------------------------------- |
| **CPU**   | 8 cores                      | 32-cores                                                   |
| **GPU**   | NVIDIA GPU (CUDA 12.6+)      | NVIDIA GPU with 24 GiB+ VRAM (Ampere or newer, CUDA 12.6+) |

The paper’s evaluation used A100 40 GB, V100 32 GB, RTX-3090 24 GB, and the corresponding host CPUs (Xeon E5-2603 v4 & Threadripper 3970X). We recommend a CPU with a higher core count as the compilation process requires building LLVM, which takes a significant amount of CPU resources.

## Getting Started
> In the Getting Started Guide, give instructions for setup and basic testing. List any software requirements and/or passwords needed to access the artifact. The instructions should take roughly 30 minutes to complete. Reviewers will follow the guide during an initial kick-the-tires phase and report issues as they arise.

> The Getting Started Guide should be as simple as possible, and yet it should stress the key elements of your artifact. Anyone who has followed the Getting Started Guide should have no technical difficulties with the rest of your artifact.

Note that we assume that the target machine has a valid CUDA and GPU driver installation with CUDA 12.6+.

0. **Install NVIDIA container toolkit**

**TODO** Say something about `NVIDIAA_CONTAINER_TOOLKIT_VERSION`

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
docker start -i constable:latest
```

3. **Run the smoke test**

```bash
JAX_PLATFORM=cpu EQSAT_PLATFORM=cpu python tests/llama.py # optimises llama on CPU; finishes in < 30 seconds
JAX_PLATFORM=gpu EQSAT_PLATFORM=gpu python tests/llama.py # optimises llama on GPU; finishes in < 30 seconds
```

Each script prints three numbers: XLA runtime, Enzyme's default optimization pipeline runtime (DefOpt) and Constable runtime. If the smoke test passes, all dependencies are correctly installed.


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
**TODO**
* Run experiment/baseline.sh
* can modify platform/models
* highlight which ones take long, and how long (would be a good idea to provide scripts that do the fast vs slow running one**
* say which CSVs we expect

#### Graphing
**TODO**
* say how to run the graphing script given the result csvs
* Should mention that we only need to pass in one run, and it attempts to find all 9 runs based on the filename

### Constable vs Enzyme-JAX
**TODO** similar content as baseline

### Cost model ablation
**TODO** similar content as baseline

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
> In the Reusability Guide, explain which parts of your artifact constitute the core pieces which should be evaluated for reusability. Explain how to adapt the artifact to new inputs or new use cases. Provide instructions for how to find/generate/read documentation about the core artifact. Articulate any limitations to the artifact’s reusability.
### Using Constable with your own JAX program
**TODO** Brief description of how to trigger the `equality-saturation` pass on an arbitrary JAX function. We can provide a template. We should also explain how to use `test_utils.py` - again we can give a template

### Extending Constable
#### File listing
**TODO** Files we have added and/or modified, and their purpose. We should explain the FFI boundaries also

