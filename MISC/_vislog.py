import re
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# def parse_training_losses(log_content):
#     losses = {
#         'loss_rpn_cls': [], 'loss_rpn_bbox': [],
#         'loss_cls': [], 'loss_bbox': []
#     }
#     epochs = []
#     curr_epoch = None
    
#     pattern = r"Epoch\(train\)\s+\[(\d+)\].*loss_rpn_cls: ([\d.]+).*loss_rpn_bbox: ([\d.]+).*loss_cls: ([\d.]+).*loss_bbox: ([\d.]+)"
    
#     for line in log_content.split('\n'):
#         match = re.search(pattern, line)
#         if match:
#             epoch = int(match.group(1))
#             if curr_epoch != epoch:
#                 curr_epoch = epoch
#                 epochs.append(epoch)
#                 losses['loss_rpn_cls'].append(float(match.group(2)))
#                 losses['loss_rpn_bbox'].append(float(match.group(3)))
#                 losses['loss_cls'].append(float(match.group(4)))
#                 losses['loss_bbox'].append(float(match.group(5)))
    
#     return np.array(epochs), losses

# def parse_validation_maps(log_content):
#     maps = {
#         'mAP': [], 'mAP_50': [], 'mAP_75': [], 
#         'mAP_s': [], 'mAP_m': [], 'mAP_l': []
#     }
#     epochs = []
#     pattern = r"bbox_mAP_copypaste: ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)"
    
#     for line in log_content.split('\n'):
#         match = re.search(pattern, line)
#         if match:
#             maps['mAP'].append(float(match.group(1)))
#             maps['mAP_50'].append(float(match.group(2)))
#             maps['mAP_75'].append(float(match.group(3)))
#             maps['mAP_s'].append(float(match.group(4)))
#             maps['mAP_m'].append(float(match.group(5)))
#             maps['mAP_l'].append(float(match.group(6)))
#             epochs.append(len(epochs) + 1)
    
#     return np.array(epochs), maps


# def plot_metrics(log_content):
#     # Get experiment name from log
#     exp_name = ""
#     for line in log_content.split('\n'):
#         if "Exp name:" in line:
#             exp_name = line.split("Exp name:")[-1].strip()
#             break
    
#     fig = plt.figure(figsize=(20, 8))
#     fig.suptitle(exp_name, fontsize=14, y=1.05)
    
#     # Plot training losses
#     ax1 = plt.subplot(1, 2, 1)
#     epochs_train, losses = parse_training_losses(log_content)
    
#     colors = {
#         'loss_rpn_cls': 'blue',
#         'loss_rpn_bbox': 'red',
#         'loss_cls': 'green',
#         'loss_bbox': 'purple'
#     }
    
#     for loss_type, values in losses.items():
#         line = ax1.plot(epochs_train, values, color=colors[loss_type], label=loss_type, marker='o', markersize=4)
#         # Add min and max value annotations
#         min_val = min(values)
#         max_val = max(values)
#         min_epoch = epochs_train[np.argmin(values)]
#         max_epoch = epochs_train[np.argmax(values)]
        
#         ax1.annotate(f'min: {min_val:.3f}', 
#                     xy=(min_epoch, min_val), 
#                     xytext=(10, 10),
#                     textcoords='offset points',
#                     color=colors[loss_type],
#                     fontsize=8)
#         ax1.annotate(f'max: {max_val:.3f}', 
#                     xy=(max_epoch, max_val), 
#                     xytext=(10, -10),
#                     textcoords='offset points',
#                     color=colors[loss_type],
#                     fontsize=8)
    
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.set_title('Training Losses')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend()
    
#     # Plot mAP metrics
#     ax2 = plt.subplot(1, 2, 2)
#     epochs_val, maps = parse_validation_maps(log_content)
    
#     colors_map = {
#         'mAP': 'blue', 'mAP_50': 'red', 'mAP_75': 'green',
#         'mAP_s': 'purple', 'mAP_m': 'orange', 'mAP_l': 'brown'
#     }
    
#     for metric, values in maps.items():
#         line = ax2.plot(epochs_val, values, color=colors_map[metric], label=f'coco/{metric}', marker='o', markersize=4)
#         # Add min and max value annotations
#         max_val = max(values)
#         max_epoch = epochs_val[np.argmax(values)]
#         ax2.annotate(f'max: {max_val:.3f}', 
#                     xy=(max_epoch, max_val), 
#                     xytext=(10, 10),
#                     textcoords='offset points',
#                     color=colors_map[metric],
#                     fontsize=8)
    
#     ax2.set_xlabel('Epoch')
#     ax2.set_ylabel('mAP Score')
#     ax2.set_title('COCO mAP Metrics')
#     ax2.grid(True, alpha=0.3)
#     ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     plt.tight_layout()
#     return plt


def parse_training_losses(log_content):
    losses = {
        'loss_rpn_cls': [], 'loss_rpn_bbox': [],
        'loss_cls': [], 'loss_bbox': []
    }
    epochs = []
    curr_epoch = None
    
    # Updated pattern to be more flexible
    pattern = r"Epoch\(train\)\s*\[(\d+)\].*loss_rpn_cls:\s*([\d.]+).*loss_rpn_bbox:\s*([\d.]+).*loss_cls:\s*([\d.]+).*loss_bbox:\s*([\d.]+)"
    
    for line in log_content.split('\n'):
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            if curr_epoch != epoch:
                curr_epoch = epoch
                epochs.append(epoch)
                losses['loss_rpn_cls'].append(float(match.group(2)))
                losses['loss_rpn_bbox'].append(float(match.group(3)))
                losses['loss_cls'].append(float(match.group(4)))
                losses['loss_bbox'].append(float(match.group(5)))
    
    if not epochs:  # Check if any matches were found
        raise ValueError("No training loss data found in log file")
        
    return np.array(epochs), losses

def parse_validation_maps(log_content):
    maps = {
        'mAP': [], 'mAP_50': [], 'mAP_75': [], 
        'mAP_s': [], 'mAP_m': [], 'mAP_l': []
    }
    epochs = []
    # Updated pattern to be more flexible
    pattern = r"bbox_mAP_copypaste:\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)"
    
    for line in log_content.split('\n'):
        match = re.search(pattern, line)
        if match:
            maps['mAP'].append(float(match.group(1)))
            maps['mAP_50'].append(float(match.group(2)))
            maps['mAP_75'].append(float(match.group(3)))
            maps['mAP_s'].append(float(match.group(4)))
            maps['mAP_m'].append(float(match.group(5)))
            maps['mAP_l'].append(float(match.group(6)))
            epochs.append(len(epochs) + 1)
    
    if not epochs:  # Check if any matches were found
        raise ValueError("No validation mAP data found in log file")
        
    return np.array(epochs), maps

def plot_metrics(log_content):
    try:
        exp_name = ""
        for line in log_content.split('\n'):
            if "Exp name:" in line:
                exp_name = line.split("Exp name:")[-1].strip()
                break
        
        fig = plt.figure(figsize=(20, 8))
        fig.suptitle(exp_name, fontsize=14, y=1.02)
        
        # Plot training losses
        ax1 = plt.subplot(1, 2, 1)
        epochs_train, losses = parse_training_losses(log_content)
        
        colors = {
            'loss_rpn_cls': 'blue',
            'loss_rpn_bbox': 'red',
            'loss_cls': 'green',
            'loss_bbox': 'purple'
        }
        
        for loss_type, values in losses.items():
            line = ax1.plot(epochs_train, values, color=colors[loss_type], label=loss_type, marker='o', markersize=4)
            min_val = min(values)
            max_val = max(values)
            min_epoch = epochs_train[np.argmin(values)]
            max_epoch = epochs_train[np.argmax(values)]
            
            ax1.annotate(f'min: {min_val:.3f}', 
                        xy=(min_epoch, min_val), 
                        xytext=(10, 10),
                        textcoords='offset points',
                        color=colors[loss_type],
                        fontsize=8)
            ax1.annotate(f'max: {max_val:.3f}', 
                        xy=(max_epoch, max_val), 
                        xytext=(10, -10),
                        textcoords='offset points',
                        color=colors[loss_type],
                        fontsize=8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot mAP metrics
        ax2 = plt.subplot(1, 2, 2)
        epochs_val, maps = parse_validation_maps(log_content)
        
        colors_map = {
            'mAP': 'blue', 'mAP_50': 'red', 'mAP_75': 'green',
            'mAP_s': 'purple', 'mAP_m': 'orange', 'mAP_l': 'brown'
        }
        
        for metric, values in maps.items():
            if all(v != -1 for v in values):  # Only plot if values are valid
                line = ax2.plot(epochs_val, values, color=colors_map[metric], label=f'coco/{metric}', marker='o', markersize=4)
                max_val = max(values)
                max_epoch = epochs_val[np.argmax(values)]
                ax2.annotate(f'max: {max_val:.3f}', 
                            xy=(max_epoch, max_val), 
                            xytext=(10, 10),
                            textcoords='offset points',
                            color=colors_map[metric],
                            fontsize=8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP Score')
        ax2.set_title('COCO mAP Metrics')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return plt
        
    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        raise

def make_plot(log_pth):
    with open(log_pth, 'r') as f:
        log_content = f.read()
    plt = plot_metrics(log_content)
    save_pth = os.path.join(os.path.dirname(log_pth), 'visualize_metrics.png')
    plt.savefig(save_pth, bbox_inches = 'tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics from log file')
    # parser.add_argument('--log_pth', default = '/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/hinton_ATTNet_r50_fpn_2x_datav2_flir_adas_rgbt_lr001/20250123_145604/20250123_145604.log', type=str, help='Path to the log file')
    parser.add_argument('--log_pth', default = '/SSDb/jemo_maeng/src/Project/Drone24/detection/drone-mmdetection-jm/work_dirs/hinton_AttNet_r50_fpn_2x_llvip_rgbt_lr005/20250114_195559/20250114_195559.log', type=str, help='Path to the log file')
    args = parser.parse_args()
    make_plot(args.log_pth)


if __name__ == '__main__':
    main()