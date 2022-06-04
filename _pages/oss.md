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

In the past few years of working at NEC Labs, I have managed to release several OSS pacakges on GitHub for ease of the deployment of a retail video surveillance POC.
The design and organization follow the Dependency Inversion Principle (DIP) instead of direct dependencies on a particular deep learning framework such as `PyTorch`.
A target ML application is supposed to invoke the APIs provided by the packages.
While the best documentation is the code per se, several feature highlights are worth going through.
Hopefully, one may find them useful for production ML applications.

## [ML](https://github.com/necla-ml/ML)

So far, the only backend is `PyTorch` and `ML` essentially mimics the APIs.
Nonetheless, it is possible to replace with other alternatives such as `TensorFlow`.
This is in case that the lastest release of `PyTorch` comes with incompatibility or even bugs.
In that regard, it allows `ML` to quickly work around those potential issues before the official fix such taht the target ML application remains intact.

<!-- Feature highlights:
- Flexible Configurations
- HDF5 Compression
- GPU Visibility
- Daemon Mode
- Distributed Training and Execution
- TensorRT Deployment
- Checkpoints from AWS/S3 and Google Drive -->

### Flexible Configurations

The configuration APIs follows the YAML format with several enhancements:

- Accept scientific notation without decimal point in case of YAML 1.1
- Support a custom YAML constructor `!include` for ease of hierarchical configuration management
  - The imported YAML config is allowed to update its parent YAML config nodes

Here is a sample `app` project with configuration files under `configs/`:
```
|____app
| |____configs
| | |____defaults.yml
| | |____sites
| | | |____site2.yml
| | | |____site1.yml
| | |____app.yml
| |____src
| | |____program.py
```

A minimal sample `program.py` is as follows to wrap its `main(...)` with the `ML` app launchar:
```py
#!/usr/bin/env python

def main(cfg):
    print(f"Running main() with cfg={cfg}")

if __name__ == '__main__':
    from ml import app
    from ml.argparse import ArgumentParser
    parser = ArgumentParser()
    cfg = parser.parse_args()
    app.run(main, cfg)
```

Sample configuration files are placed under `configs/`.
The top level one is `configs/app.yml` for the program to invoke with option `--cfg`:

```sh
$> ./program.py --cfg configs/app.yml
```

Looking into `configs/app.yml` below, a custom `!include` constructor is supported to include a specific configuration file that updates its parent configurations.

```yaml
import:
  defaults: 
    app_site: !include defaults.yml
  app_site: !include sites/site1.yml

app_platform: Apple M1
app_OS: MacOSX
```

`configs/defaults.yml` serves as the default site config as follows:

```yaml
  name: Little Planet Branch
  location: St. Louis, MO
  template: XXX
```

`configs/sites/site1.yml` is the configuration for a specific site that may overwrite the defualts:

```yaml
  name: site1
  location: Santa Clara, CA
  note: Welcome to CA
```

Since `app.yml` includes `site1.yml`, `site1.yml` will update the configuration in `defaults.yml` as updating a `Python` dictionary.
Therefore, the resulting configuration is as follows:

```sh
Running main() with cfg={   '__file__': PosixPath('configs/app.yml'),
    'app_OS': 'MacOSX',
    'app_platform': 'Apple M1',
    'app_site': {   'location': 'Santa Clara, CA',
                    'name': 'site1',
                    'note': 'Welcome to CA',
                    'template': 'XXX'},
    'daemon': False,
    'deterministic': False,
    'dist': None,
    'dist_backend': 'nccl',
    'dist_no_sync_bn': False,
    'dist_port': 25900,
    'dist_url': 'env://',
    'gpu': [],
    'logfile': None,
    'logging': False,
    'no_gpu': False,
    'rank': -1,
    'seed': 1204,
    'slurm_constraint': None,
    'slurm_cpus_per_task': 4,
    'slurm_export': '',
    'slurm_mem': '32G',
    'slurm_nodelist': None,
    'slurm_nodes': 1,
    'slurm_ntasks_per_node': 1,
    'slurm_partition': 'gpu',
    'slurm_time': '0',
    'world_size': -1}
```

Other key-value settings are specified by `ML` for other features.
Note only keys `name` and `location` are replced while `template` remains as is.
The semantics for additional keys such as `note` is to merged with the parent configuration.

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

TBD

## [ML-WS](https://github.com/necla-ml/ML-WS)

TBD
## [feedstocks](https://github.com/necla-ml/feedstocks)

TBD