import open3d as o3d
import os
import sys
import glob
import numpy as np
import cv2

def process_pcd(pcd_path):
    """
    Process PCD file and save visualization without displaying.
    Returns the RGB image as a numpy array.
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # Create a visualizer object in headless mode
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Create invisible window
    
    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)
    
    # Get render options and set background color
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # Black background
    
    # Set view control for desired view
    ctr = vis.get_view_control()
    ctr.set_front([-5, 0, -1])  # Looking down along z-axis
    ctr.set_up([0, 0.5, 0])     # Up direction is y-axis
    ctr.set_lookat([0, 0, 0])   # Look at center
    ctr.set_zoom(0.1)           # Adjust zoom level
    
    # Update visualization
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    # Capture the screen as a numpy array
    image_float_buffer = vis.capture_screen_float_buffer(do_render=True)
    rgb_image = (np.asarray(image_float_buffer) * 255).astype(np.uint8)
    rgb_image = rgb_image[:,480:-480,:]
    rgb_image = cv2.resize(rgb_image, (120, 160))
    print(rgb_image.shape)
    cv2.imwrite('test.png', rgb_image)
    # Destroy the visualizer
    vis.destroy_window()
    
    return rgb_image
    # Clean up

def process_sequence(pcd_dir, output_dir):
    """
    Process a sequence of PCD files without visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PCD files
    pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
    
    if not pcd_files:
        print(f"No PCD files found in {pcd_dir}")
        return
    
    # Process each PCD file
    for i, pcd_path in enumerate(pcd_files):
        # Generate output filename
        output_path = os.path.join(output_dir, f"{i:04d}.png")
        
        # Process and save
        process_pcd(pcd_path, output_path)
        print(f"Processed {os.path.basename(pcd_path)} -> {os.path.basename(output_path)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viz_pcd.py <pcd_file_or_directory> <output_directory>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    
    if os.path.isfile(input_path):
        process_pcd(input_path)
    else:
        print(f"Invalid path: {input_path}")
