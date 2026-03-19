import numpy as np
from PIL import Image
import random
import cv2
import os

def calculate_fitness(img_array, ref_array):
    score = 0
    # scales: 1 (pixel) to 16 (patches - details to big shapes)
    # Weights: We give more 'importance' to larger scales to help big shapes form first
    scales = [1, 2, 4, 8, 16, 32]
    weights = [1, 2, 4, 4, 2, 1]
    
    for s, w in zip(scales, weights):
        if s == 1:
            diff = np.abs(img_array - ref_array)
        else:
            diff = np.abs(img_array[::s, ::s] - ref_array[::s, ::s])
        score += np.sum(diff) * w
    return score

def get_gray(pixels):
    return 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]

def run_evolution(start_path, goal_path, width=180, height=120, output_img="final_result.jpg", output_video="pixel_dance.mp4", iterations=150000):
    width, height = 240, 160 

    if not os.path.exists(start_path) or not os.path.exists(goal_path):
        print("Error: Make sure your image files are in the same folder as this script!")
        return

    ref_img = Image.open(goal_path).convert("RGB").resize((width, height))
    alt_img = Image.open(start_path).convert("RGB").resize((width, height))
    
    curr_pixels = np.array(alt_img).astype(np.float32)
    target_gray = np.array(ref_img.convert("L")).astype(np.float32)
    current_fitness = calculate_fitness(get_gray(curr_pixels), target_gray)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    video_writer = cv2.VideoWriter(output_video, fourcc, 60.0, (width, height))
    
    #simulated anealing parameters
    temp = 1.0
    cooling_rate = 0.99998 
    
    print(f"Starting SA Evolution...")
    print("Commands: [q] to save and quit | [s] to save current frame")

    for i in range(iterations + 1):
        y1, x1 = random.randint(0, height-1), random.randint(0, width-1)
        y2, x2 = random.randint(0, height-1), random.randint(0, width-1)
        
        p1, p2 = curr_pixels[y1, x1].copy(), curr_pixels[y2, x2].copy()
        curr_pixels[y1, x1], curr_pixels[y2, x2] = p2, p1
        
        new_fitness = calculate_fitness(get_gray(curr_pixels), target_gray)
        
        if new_fitness < current_fitness or random.random() < np.exp((current_fitness - new_fitness) / temp):
            current_fitness = new_fitness
        else:
            curr_pixels[y1, x1], curr_pixels[y2, x2] = p1, p2
        
        temp *= cooling_rate
        
        if i % 500 == 0:
            frame_bgr = cv2.cvtColor(curr_pixels.astype(np.uint8), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            preview = cv2.resize(frame_bgr, (480, 320), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Pixel Evolution (SA)", preview)
            
            if i % 10000 == 0:
                print(f"Step {i} | Error: {int(current_fitness)} | Temp: {temp:.4f}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'): cv2.imwrite(f"snapshot_{i}.jpg", frame_bgr)

    video_writer.release()
    cv2.destroyAllWindows()
    Image.fromarray(curr_pixels.astype(np.uint8)).save(output_img)
    print(f"Success! Video saved as {output_video}")

if __name__ == "__main__":
    run_evolution(start_path='start.jpg', goal_path='goal.jpg', 
                  width=240, height=160, 
                  output_img="final_result.jpg",
                  output_video="pixel_dance.mp4", 
                  iterations=300000)