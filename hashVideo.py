import cv2
from os import listdir
from os.path import isfile, join
import time,sys
import pickle
import imagehash
from PIL import Image

# Perceptual hash algorithm. LSH for frames
def phash(image,hashSize=4):
    # Return a truth matrix, a matrix of boolean values
    diff = imagehash.phash(Image.fromarray(image),hash_size=hashSize).hash
    # Flatten and convert boolean values to x-bit number
    sums = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    return sums

# Wavelet hash algorithm. LSH for frames
def whash(image,hashSize=4):
    diff = imagehash.whash(Image.fromarray(image),hash_size=hashSize).hash
    return  sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

# Function to compute feature vector for a given video
def get_features(name,num_frames,hash_size):
    path = 'videos/'+name # relative path to video file
    cap = cv2.VideoCapture(path) # OpenCV object to get frames of video
    split = int(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / num_frames) # compute the interval between each key frame

    feature_vector = []
    for frame_no in range(0,num_frames*split,split): # frame_no is the index of the frame in the video.
        cap.set(1, frame_no) # Move the cursor to the next frame
        feature_vector.append(phash(cap.read()[1],hashSize=hash_size)) # Read frame, hash it and append to output vector

    cap.release() # Release the lock
    return feature_vector

# Search through the data directory and return all filenames
def get_video_names():
    return [f for f in listdir('videos/') if isfile(join('videos/', f)) and '.mp4' in f]

# Measure the time used for hashing
t0 = time.time()

# PARAMETERS FOR VIDEO HASHING ALGORITHM
num_features = 6 # = number of key frames to be chosen = length of feature vector
hash_size = 6 # hash_size ** 2 is the number of bits in the hash value. Resizing parameter
filenames = get_video_names()
length = len(filenames)

store = {} # Dictionary to store on disk
for idx,file in enumerate(filenames): # For each video
    if idx % 2 == 0:
        # Print progress every once in a while
        sys.stdout.write("\r\t%.2f%% processed" % (float(idx+1)/float(length)*100))
        sys.stdout.flush()
    # Store the features vector as value to the file as key
    store[file] = get_features(file,num_features,hash_size)

# All hail OpenCV
cv2.destroyAllWindows()

# Store the hashed videos on disk for later use.
with open('data_'+str(num_features)+'_phash_'+str(hash_size**2)+'.p', 'wb') as fp:
    pickle.dump(store, fp)

# Report time
print("\n\n%.2f seconds" % (time.time() - t0))