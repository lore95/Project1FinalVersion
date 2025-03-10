import os
import itertools
from PIL import Image


def MergePics(source_dir, target_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    joint_keywords = ['Ankle', 'Hip', 'Knee', 'Pelvis', 'Trunk']

    def find_matching_keyword(filename, keywords):
        """Finds the first matching keyword in a filename."""
        for keyword in keywords:
            if keyword in filename:
                return keyword
        return None

    # Get list of PNG files in source directory
    source_files = [f for f in os.listdir(source_dir) if f.endswith(".png")]
    target_files = [f for f in os.listdir(target_dir) if f.endswith(".png")]

    # Group files by joint keyword
    source_files_by_joint = {}
    target_files_by_joint = {}

    for file in source_files:
        joint = find_matching_keyword(file, joint_keywords)
        if joint:
            source_files_by_joint[joint] = os.path.join(source_dir, file)

    for file in target_files:
        joint = find_matching_keyword(file, joint_keywords)
        if joint:
            target_files_by_joint[joint] = os.path.join(target_dir, file)

    # Merge images vertically
    for joint, source_path in source_files_by_joint.items():
        if joint in target_files_by_joint:
            target_path = target_files_by_joint[joint]
            
            # Open images
            img1 = Image.open(target_path)
            img2 = Image.open(source_path)
            
            # Ensure the images have the same width
            width = max(img1.width, img2.width)
            img1 = img1.resize((width, img1.height))
            img2 = img2.resize((width, img2.height))
            
            # Create a new image with combined height
            merged_img = Image.new('RGB', (width, img1.height + img2.height))
            
            # Paste images on top of each other
            merged_img.paste(img1, (0, 0))
            merged_img.paste(img2, (0, img1.height))
            
            # Save the merged image
            output_path = os.path.join(output_dir, f"Merged_{joint}.png")
            merged_img.save(output_path)
            print(f"Saved merged image: {output_path}")

def merge_3images_vertically(source_dir, target_dir, third_dir, output_dir):
    """Merge images vertically if they exist in all three directories."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PNG files in source directory
    source_files = {f for f in os.listdir(source_dir) if f.endswith(".png")}
    target_files = {f for f in os.listdir(target_dir) if f.endswith(".png")}
    third_files = {f for f in os.listdir(third_dir) if f.endswith(".png")}
    
    # Find common files
    common_files = source_files.intersection(target_files, third_files)
    
    for file_name in common_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        third_path = os.path.join(third_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        # Open images
        img1 = Image.open(source_path)
        img2 = Image.open(target_path)
        img3 = Image.open(third_path)
        
        # Ensure the images have the same width
        width = max(img1.width, img2.width, img3.width)
        img1 = img1.resize((width, img1.height))
        img2 = img2.resize((width, img2.height))
        img3 = img3.resize((width, img3.height))
        
        # Create a new image with combined height
        merged_img = Image.new('RGB', (width, img1.height + img2.height + img3.height))
        
        # Paste images on top of each other
        merged_img.paste(img3, (0, 0))
        merged_img.paste(img2, (0, img3.height))
        merged_img.paste(img1, (0, img3.height + img2.height))
        
        # Save the merged image
        merged_img.save(output_path)
        print(f"Merged {file_name} and saved to {output_path}")

def merge_3images_horizontally(source_dir, target_dir, third_dir, output_dir):
    """Merge images horizontally if they exist in all three directories."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PNG files in source directory
    source_files = {f for f in os.listdir(source_dir) if f.endswith(".png")}
    target_files = {f for f in os.listdir(target_dir) if f.endswith(".png")}
    third_files = {f for f in os.listdir(third_dir) if f.endswith(".png")}
    
    # Find common files
    common_files = source_files.intersection(target_files, third_files)
    
    for file_name in common_files:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        third_path = os.path.join(third_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        # Open images
        img1 = Image.open(source_path)
        img2 = Image.open(target_path)
        img3 = Image.open(third_path)
        
        # Ensure the images have the same height
        height = max(img1.height, img2.height, img3.height)
        img1 = img1.resize((img1.width, height))
        img2 = img2.resize((img2.width, height))
        img3 = img3.resize((img3.width, height))
        
        # Create a new image with combined width
        merged_img = Image.new('RGB', (img1.width + img2.width + img3.width, height))
        
        # Paste images side by side
        merged_img.paste(img1, (0, 0))
        merged_img.paste(img2, (img1.width, 0))
        merged_img.paste(img3, (img1.width + img2.width, 0))
        
        # Save the merged image
        merged_img.save(output_path)
        print(f"Merged {file_name} and saved to {output_path}")

# Define paths
source_dir = "../plots/JoggingJointAngles"
target_dir = "../plots/JointAngles"
output_dir = "../plots/MergedJointAngles"

MergePics(source_dir, target_dir,output_dir)

# Define paths
source_dir = "../plots/JogMoments"
target_dir = "../plots/Moments"
output_dir = "../plots/MergedMoments"

MergePics(source_dir, target_dir,output_dir)


# Define paths
source_dir = "../plots/JogPowers"
target_dir = "../plots/Powers"
output_dir = "../plots/MergedPowers"

MergePics(source_dir, target_dir,output_dir)


source_dir = "../plots/MergedPowers"
target_dir = "../plots/MergedMoments"
third_dir =  "../plots/MergedJointAngles"


output_dir = "../plots/FullyMerged"


merge_3images_horizontally(third_dir, target_dir,source_dir,output_dir)


