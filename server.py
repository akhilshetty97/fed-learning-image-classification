import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from CNNmodel import create_model 

def server_fn():
    # Initialize the global model
    model = create_model() 
    parameters = ndarrays_to_parameters(model.get_weights())

    #  Initialize the strategy
    strategy = FedAvg(
        fraction_fit=1.0,            
        fraction_evaluate=1.0,       
        min_available_clients=2,    
        initial_parameters=parameters 
    )
    return strategy

# Start the Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080", 
    strategy=server_fn(),  
    config=fl.server.ServerConfig(num_rounds=3)
)
