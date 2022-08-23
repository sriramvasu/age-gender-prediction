import tensorflow as tf
from tensorflow.python.tools import freeze_graph

def convert_saved_model_to_pb(input_saved_model_dir, output_graph_dir):
    # output_node_names = ','.join(output_node_names)
    freeze_graph.freeze_graph(input_graph=None, input_saver=None,
                              input_binary=None,
                              input_checkpoint=None,
                              output_node_names=None,
                              restore_op_name=None,
                              filename_tensor_name=None,
                              output_graph=output_graph_dir,
                              clear_devices=None,
                              initializer_nodes=None,
                              input_saved_model_dir=input_saved_model_dir)


def save_output_tensor_to_pb():
    # output_names = ['StatefulPartitionedCall']
    save_pb_model_path = './vggface_frozen.pb'
    model_dir = './vggface_tensorflow.pb'
    convert_saved_model_to_pb(model_dir, save_pb_model_path)
    # print(graph.get_operations())
save_output_tensor_to_pb()