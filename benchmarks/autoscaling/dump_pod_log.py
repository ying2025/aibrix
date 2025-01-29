from kubernetes import client, config
import sys
import os
import time

def get_pods(deployment_name, namespace):
    config.load_kube_config(context="ccr3aths9g2gqedu8asdg@41073177-kcu0mslcp5mhjsva38rpg")
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=namespace)
    return [pod for pod in pods.items if deployment_name in pod.metadata.name]

def dump_logs(pod, pod_logs_dir, namespace, container_name):
    v1 = client.CoreV1Api()
    try:
        logs = v1.read_namespaced_pod_log(
            name=pod.metadata.name,
            namespace=namespace,
            container=container_name,
            timestamps=True
        )
        
        try:
            previous_logs = v1.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
                container=container_name,
                previous=True,
                timestamps=True
            )
            if previous_logs:
                logs = "=== Previous Container Logs ===\n" + previous_logs + "\n=== Current Container Logs ===\n" + logs
        except:
            pass

        if logs:
            filename = f"{pod_logs_dir}/{pod.metadata.name}-{container_name}-{int(time.time())}.log"
            os.makedirs(pod_logs_dir, exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write(logs)
            print(f"Saved logs for {pod.metadata.name} container {container_name}")
        else:
            print(f"No logs found for {pod.metadata.name} container {container_name}")
            
    except Exception as e:
        print(f"Error getting logs for {pod.metadata.name} container {container_name}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python dump_pod_log.py <deployment_name> <pod_logs_dir> <namespace> <container_name>")
        sys.exit(1)

    deployment_name = sys.argv[1]
    pod_logs_dir = sys.argv[2]
    namespace = sys.argv[3]
    container_name = sys.argv[4]

    pods = get_pods(deployment_name, namespace)
    if not pods:
        print("ERROR: No pods found")
        sys.exit(1)
        
    print(f"dump_pod_log, Found {len(pods)} pods for deployment {deployment_name}")
    for pod in pods:
        dump_logs(pod, pod_logs_dir, namespace, container_name)












# from kubernetes import client, config
# import sys
# import os

# def get_pods(deployment_name, namespace):
#     config.load_kube_config(context="ccr3aths9g2gqedu8asdg@41073177-kcu0mslcp5mhjsva38rpg")
#     v1 = client.CoreV1Api()
#     pods = v1.list_namespaced_pod(namespace=namespace)
#     return [pod for pod in pods.items if deployment_name in pod.metadata.name]

# def dump_logs(pod, pod_logs_dir, namespace, container_name):
#     v1 = client.CoreV1Api()
#     try:
#         logs = v1.read_namespaced_pod_log(
#             name=pod.metadata.name,
#             namespace=namespace,
#             container=container_name
#         )
        
#         filename = f"{pod_logs_dir}/{pod.metadata.name}-{container_name}.log"
#         os.makedirs(pod_logs_dir, exist_ok=True)
        
#         with open(filename, 'w') as f:
#             f.write(logs)
#         print(f"Saved logs for {pod.metadata.name} container {container_name}")
        
#     except Exception as e:
#         print(f"Error getting logs for {pod.metadata.name} container {container_name}: {e}")

# if __name__ == "__main__":
#     if len(sys.argv) != 5:
#         print("Usage: python dump_pod_log.py <deployment_name> <pod_logs_dir> <namespace> <container_name>")
#         sys.exit(1)

#     deployment_name = sys.argv[1]
#     pod_logs_dir = sys.argv[2]
#     namespace = sys.argv[3]
#     container_name = sys.argv[4]

#     pods = get_pods(deployment_name, namespace)
#     if not pods:
#         print("ERROR: No pods found")
#         sys.exit(1)
        
#     print(f"dump_pod_log, Found {len(pods)} pods for deployment {deployment_name}")
#     for pod in pods:
#         dump_logs(pod, pod_logs_dir, namespace, container_name)