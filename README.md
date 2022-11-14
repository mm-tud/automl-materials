# AutoML Benchmark for Regression Tasks on Small Tabular Data 

The code in this repository accompanies the paper [“Benchmarking AutoML for regression tasks on small tabular data in materials design“](https://doi.org/10.1038/s41598-022-23327-1).

Bibtex entry:

```
@article{Conrad2022,
    author  = {Conrad, Felix and M{\"a}lzer, Mauritz and Schwarzenberger, Michael and Wiemer, Hajo and Ihlenfeldt, Steffen},
    title   = {Benchmarking AutoML for regression tasks on small tabular data in materials design},
    journal = {Scientific Reports},
    year    = {2022},
    month   = {Nov},
    day     = {11},
    volume  = {12},
    number  = {1},
    pages   = {19350},
    issn    = {2045-2322},
    doi     = {10.1038/s41598-022-23327-1},
    url     = {https://doi.org/10.1038/s41598-022-23327-1}
}
```

## Getting Started 

A basic linux machine with an installation of docker is able to run the benchmark.
The start scripts (`start.sh` and `run.py` for each framework) assume at least 8 CPU cores. 
This "requirement" can be softened by editing the `Dockerfile`s and `run.py`s.

Excute the `run_all.sh` script in the base directory to launch all frameworks with “our“ default settings. 
To launch only single frameworks navigate to `framework/chosen-framework` and execute `start.sh`.

*Note: If run for the first time, a login to DockerHub might be necessary in order not to exceed the rate limits for image pulls. (`docker login -u username -p $(cat docker_access_token)`)*

## Datasets and License

Only a subset of the datasets used in our publication is included in this repository to preserve copyright.
The subset is sufficient as minimal working example.
It can be used to test the code and it displays the required data structure for the framework.
The subset contains a slightly formatted version of `UCI-conrete` and `Yin-2021`: 

* `UCI-concrete` 
    - full name: "Concrete Compressive Strength Data Set"
    - available from https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    - reuse of the database is unlimited with retention of copyright notice
    - copyright by Prof. I-Cheng Yeh
    - introduced in "I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)."
* `Yin-2021`
    - published as attachment to "Yin, B. B., and K. M. Liew. "Machine learning and materials informatics approaches for evaluating the interfacial properties of fiber-reinforced composites." Composite Structures 273 (2021): 114328."
    - available on github without further specification of a license: https://github.com/Binbin202/ML-Data

The remaining datasets used in our publication can be obtained from the respective owners upon reasonable request or by accessing the original publications. 
