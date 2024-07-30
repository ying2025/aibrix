
# Notes
Aggregated Discovery Service enables envoy to dynamically register upstream clusters and routes

There are some issues in enabling ext_proc for aggregated discovery service as envoyExtensionPolicy, so for now directly starting envoy as a standalone service.

Containerize envoy and gateway.


# Install test model service
```
cd docs/tutorial/m1
pip install --no-cache-dir -r requirements.txt
python app.py
```


Start ext_proc to register model service (right now static value, use redis backend) and envoy
```
cd pkg/gateway
make run-control-plane

Install Envoy: https://www.envoyproxy.io/docs/envoy/latest/start/install#install-envoy-on-debian-gnu-linux
make run-envoy 
```