This example demonstrates how to implement a Federated Learning system for handwriting recognition using CNN, RNN, and CTC loss with the Flower framework.

python -m backend.main --mode fl_server

python -m backend.federated_learning.flwr_client --client_id 0 --learning_rate 0.01 --batch_size 64
python -m backend.federated_learning.flwr_client --client_id 1 --learning_rate 0.01 --batch_size 64

set PYTHONPATH=%PYTHONPATH%;D:\WorkSpace\HUET Graduation\source\huet_graduation\backend
