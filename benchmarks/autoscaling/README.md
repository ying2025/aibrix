# How to run autoscaling experiment

You can run autoscaling benchmark experiment by simply running the command below.

`./overnight_run.sh workload/workload/25min_up_and_down/25min_up_and_down.jsonl`

What you have to take a look
- run deployment for your application (refer to `deepseek-llm-7b-chat-v100/deploy.yaml`)
- change the name field under scaleTargetRed in all autoscaling yaml files.
- check the deployment name in run-test.py

For example,
```yaml
...
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: deepseek-llm-7b-chat-v100 (*this one)
...
```

### Process
`overnight_run.sh` will execute run-test.sh with given autoscaling mechanism.

`run-test.sh` will execute 
1. Preparation
    - clean up existing autoscaler resources
    - set the number of replicas to 1 of the target deployment
    - restart aibrix-controller-manager, aibrix-gateway-plugins, target application's deployment
2. Ready state check
    - Once all the deployments are in ready state, it will start experiment
3. Start logging running as background processes
    - Log application pods, aibrix-controller-manager, aibrix-gateway-plugins by streaming the output to files (it will be used for debugging, post analysis)
4. Run client with appropriate arguments (client code is in `aibrix/benchmark/generator/client.py`)
5. Stop logging processes once experiment is done
6. clean up resources
    - clean up autocaling resources
    - set the number of replicas for the application deployment back to 1.


### Dependency
- Client: `aibrix/benchmark/generator/client.py` 
- Workload trace generator: `aibrix/benchmark/generator/workload_generator.py`. Please check aibrix/benchmark/generator/README.md for more information.