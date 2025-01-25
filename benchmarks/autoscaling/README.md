# How to run autoscaling experiment

Before running an experiment, you need to configure below arguments in run-test.sh
- input_workload_path
- aibrix_repo path
- api_key
- k8s_config_dir
- target_deployment


After that, to run experiment, run 

`./run-test.sh workload/original/20s.jsonl &> output.txt`