import sys
from pathlib import Path
sys.path.insert(0, str(Path('.')))

import torch
from aura.fl_client import AURAFlowerClient, model_to_ndarrays, ndarrays_to_model
from aura.fl_server import get_federated_parameters
from aura.models import AURAModelBundle, FlowAutoencoder, AuraSTGNN, AttackHead
import config as cfg

def count_tensors(module):
    return len(list(module.parameters()))

def main():
    print("==================================================")
    print("TASK 3 — Parameter Inventory")
    print("==================================================")
    
    ae = FlowAutoencoder()
    stgnn = AuraSTGNN()
    head = AttackHead()
    bundle = AURAModelBundle()
    
    ae_count = count_tensors(ae)
    stgnn_count = count_tensors(stgnn)
    head_count = count_tensors(head)
    bundle_count = count_tensors(bundle)
    
    print(f"Autoencoder: {ae_count} tensors")
    print(f"STGNN: {stgnn_count} tensors")
    print(f"AttackHead: {head_count} tensors")
    print(f"Entire bundle: {bundle_count} tensors")
    print(f"Sum of parts: {ae_count + stgnn_count + head_count} tensors")
    
    # Client count
    dummy_data = torch.zeros((1, cfg.FEATURE_DIM))
    client = AURAFlowerClient(client_id="audit_client", train_data=dummy_data, val_data=dummy_data)
    client_count = len(list(client.model.parameters()))
    print(f"Flower client (client.model.parameters()): {client_count} tensors")
    
    print("\n==================================================")
    print("TASK 4 — Runtime Serialization Audit")
    print("==================================================")
    
    print(f"Immediately before serialization (client.model): {client_count}")
    
    exported_ndarrays = model_to_ndarrays(client.model)
    print(f"Immediately after serialization (model_to_ndarrays): {len(exported_ndarrays)}")
    
    # Simulate deserialization on server
    received_parameters = exported_ndarrays
    print(f"Immediately after deserialization (received_parameters): {len(received_parameters)}")
    
    # Simulate global model
    global_model = AURAModelBundle()
    global_arrays = [p.detach().cpu().numpy() for p in get_federated_parameters(global_model)]
    print(f"Immediately before aggregation (get_federated_parameters(global_model)): {len(global_arrays)}")
    
    print("\n==================================================")
    print("TASK 5 — Tensor Identity")
    print("==================================================")
    
    print("Missing tensors (in client export compared to global_model):")
    # Identify which are missing
    client_names = [name for name, _ in client.model.autoencoder.named_parameters()] + \
                   [name for name, _ in client.model.attack_head.named_parameters()]
    global_names = [name for name, _ in global_model.named_parameters() 
                    if "stgnn" not in name]
    
    for i, name in enumerate(global_names):
        if name not in [f"autoencoder.{n}" for n in [name for name, _ in ae.named_parameters()]] + \
                       [f"attack_head.{n}" for n in [name for name, _ in head.named_parameters()]]:
            p = dict(global_model.named_parameters())[name]
            print(f"  Missing: {name}, Shape: {p.shape}, Index in global: {i}")

    print("\n==================================================")
    print("TASK 6 — Shape Audit")
    print("==================================================")
    
    for i in range(min(len(exported_ndarrays), len(global_arrays))):
        c_shape = exported_ndarrays[i].shape
        g_shape = global_arrays[i].shape
        if c_shape != g_shape:
            print(f"Mismatch at index {i}!")
            # Determine which parameter this maps to in global model
            g_name = list(global_model.named_parameters())[i][0]
            print(f"  Global Name: {g_name}")
            print(f"  Client Shape: {c_shape}, dtype: {exported_ndarrays[i].dtype}")
            print(f"  Server Shape: {g_shape}, dtype: {global_arrays[i].dtype}")
            break
            
if __name__ == '__main__':
    main()
