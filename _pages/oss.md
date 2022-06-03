---
layout: page
title: OSS
permalink: /oss/
---

<style type="text/css">
.image-left {
  display: block;
  margin-top: 5px;  
  margin-right: 15px;
  margin-bottom: 0px;
  float: left;
}
</style>

In the past few years of working at NEC Labs, I have managed to release several OSS repos on GitHub for ease of the deployment of our retail video surveillance POC.
Here is a list of relevant repos to go through.
Hopefully, one may find them useful for production ML applications.

## [ML](https://github.com/necla-ml/ML)

Feature highlights:
- Flexible Configurations
- HDF5 Compression
- GPU Visibility
- Daemon Mode
- Distributed Training and Execution
- TensorRT Deployment
- Checkpoints from AWS/S3 and Google Drive

### Flexible Configurations

The configuration APIs are based on YAML with several enhancements:
- Accept scientific notation without decimal point in case of YAML 1.1
- Support YAML imports for ease of hierarchical configuration management
	- The imported YAML config is allowed to update its parent YAML config nodes 

### HDF5 Compression

When saving Python objects, depending on the suffixes, `pickle`, `h5` and `pytorch` binaries are supported.
If `h5` is chosen to save sparse binary features, compression options such as `zstd` can be enabled to reduce storage significantly.

### GPU Visibility

Deep learning programs tend to access GPUs in parallel.
Managing GPU visibility is crucial to facilitate distributed training and other processing that require GPU access.
`ML` provides simple app launcher APIs to support common GPU access options.

### Daemon Mode

It is common to kick start a long running training process through a remote shell.
Instead of figuring out how `screen` and `nohup` work, turning a process into a daemon to detach from the terminal is as simple as providing a command line option.
The `ML` app launcher APIs deal with the underlying complexities as usual as running the same program in the foreground.

### Distributed Training and Execution

Composing a distributed parallel program in Python can be daunting and error prone.
The `ML` app launcher APIs support `PyTorch` and `SLURM` backends for training and general execution given command line options.
The `PyTorch` backend assumes one GPU per worker process on one GPU node.
The `SLRUM` backend supports a cluster environment and execution across multiple GPU nodes.

### TensorRT Deployment

For production inference to be competitie, ML model deployment optimization is necessary to reduce the runtime cost.
`TensorRT` is a popular backend for `ML` to support.
The APIs make it straighforward to convert a pretrained model into its TensorRT counterpart.

### Checkpoints from AWS/S3 and Google Drive

By default, `PyTorch` hub APIs only supports loading checkpoints from direct URLs.
`ML` hub APIs further supports AWS/S3 and Google Drive for private or 3rd party checkpoint storage.
This makes it easy for business deployment at a low cost.

## [ML-Vision](https://github.com/necla-ml/ML-Vision)
## [ML-WS](https://github.com/necla-ml/ML-WS)
## [feedstocks](https://github.com/necla-ml/feedstocks)

