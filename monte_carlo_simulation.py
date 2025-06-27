from pipeline.training import MonteCarloSimulation
from pipeline.utility import  manifest_generator_wrapper
from multiprocessing import Process
from typing import List


def monte_carlo_wrapper(model_path: str, data, species_labels, species_composition, num_runs: int=4, sample_size: int=1000):
    benchmark = MonteCarloSimulation(
        model_path=model_path,
        data_manifest=data,
        dataset_species_labels=species_labels,
        is_inception_v3=False,
        # input_size=(160, 160),
        providers=["CPUExecutionProvider"],
        
    )
    benchmark.run(species_labels, species_composition, num_runs=num_runs, sample_size=sample_size, save_path="./", enable_confusion_matrix=True)


def group_launcher(processes, dominant_threshold: float, file_list: List[str]):
    all_images, _, _, species_dict, species_composition =  manifest_generator_wrapper(dominant_threshold)
    for model in file_list:
        process = Process(target=monte_carlo_wrapper, args=(model, all_images, species_dict, species_composition))
        processes.append(process)
        process.start()

if __name__ == "__main__":
    # file_list_50 = [
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_5.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_10.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_15.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_20.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_25.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_30.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_35.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_40.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_45.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_50_prune_50.onnx",
    # ]
    # file_list_80 = [
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_5.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_10.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_15.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_20.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_25.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_30.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_35.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_40.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_45.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_80_prune_50.onnx",
    # ]
    # file_list_90 = [
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_5.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_10.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_15.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_20.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_25.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_30.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_35.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_40.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_45.onnx",
    #     "/home/tom-maverick/Documents/Final Results/MobileNetV3-modified_train_prune_retrain_multiple_thresholds_layer_by_layer/mobilenet_v3_large_90_prune_50.onnx",
    # ]
    file_list = [
        # "./models/mobilenet_v3_large_50_quantized.onnx",
        # "/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.onnx"
        # "/home/tom-maverick/Documents/Final Results/InceptionV3_HG_onnx/inceptionv3_inat_other_50.onnx",
        # "/home/tom-maverick/Documents/02_GitHub/mcunet/models/mcunet_haute_garonne_10_species.onnx"
        "/home/tom-maverick/Documents/Final Results/MobileNetV3/mobilenet_v3_large_50.onnx"

    ]
    processes: List[Process] = []
    
    # group_launcher(processes, 0.5, file_list_50)
    # for process in processes:
    #     process.join()   
    # processes.clear()
    # group_launcher(processes, 0.8, file_list_80)
    # for process in processes:
    #     process.join()   
    # processes.clear()
    # group_launcher(processes, 0.9, file_list_90)
    # for process in processes:
    #     process.join()   
    # processes.clear()
    group_launcher(processes, 0.5, file_list)
    for process in processes:
        process.join()   
    processes.clear()