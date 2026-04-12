"""
SIMPLE SIMILARITY THRESHOLD APPROACH FOR TRACK MERGING

This module implements a straightforward approach:
1. Find the track with the most frames (anchor = target speaker)
2. Extract one embedding from the anchor track
3. For each other track, extract an embedding and compute similarity to anchor
4. Any track similar enough (above threshold) gets merged with anchor
5. Return the merged track

No clustering, no fragmentation detection, no secondary heuristics.
Just direct similarity comparison to a known reference point.
"""

import numpy as np
import glob
import os
import cv2
import torch
from scipy.spatial.distance import euclidean


class SimilarityThresholdTrackMerger:
    """
    Merge tracks using embedding similarity to an anchor track.
    
    The anchor track is the one with the most frames (highest confidence
    that it contains the target speaker).
    """
    
    def __init__(self, allTracks, frames_path):
        """
        Args:
            allTracks: List of tracks from track_shot()
            frames_path: Path to directory containing frame images
        """
        self.allTracks = allTracks
        self.frames_path = frames_path
    
    def find_anchor_track(self):
        """
        Find the track with the maximum number of frames.
        
        This track is almost certainly your target speaker because the
        target speaker appears throughout most of the video while
        background people appear sporadically.
        
        Returns:
            Tuple of (anchor_track_idx, num_frames_in_anchor)
        """
        max_frames = 0
        anchor_idx = None
        
        for track_idx, track in enumerate(self.allTracks):
            num_frames = len(track['frame'])
            if num_frames > max_frames:
                max_frames = num_frames
                anchor_idx = track_idx
        
        return anchor_idx, max_frames
    
    def extract_embedding(self, track_idx, embedding_model, device):
        """
        Extract a single face embedding from a track's middle frame.
        
        We sample the middle frame because it's usually the clearest part
        of the face (less likely to be at the boundary of the track where
        motion might blur the face).
        
        Args:
            track_idx: Index of the track to extract from
            embedding_model: Pre-loaded InceptionResnetV1 model
            device: torch device (cuda or cpu)
        
        Returns:
            512-dimensional embedding vector as numpy array
        """
        track = self.allTracks[track_idx]
        frames = track['frame']
        bboxes = track['bbox']
        
        # Sample from middle of track
        middle_idx = len(frames) // 2
        frame_num = int(frames[middle_idx])
        bbox = bboxes[middle_idx]
        
        # Load the frame image
        flist = sorted(glob.glob(os.path.join(self.frames_path, "*.jpg")))
        frame = cv2.imread(flist[frame_num])
        
        if frame is None:
            raise ValueError(f"Could not load frame {frame_num}")
        
        # Extract face region using bounding box
        x1, y1, x2, y2 = [int(v) for v in bbox]
        padding = 5  # Small padding around face improves embedding quality
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0:
            raise ValueError(f"Invalid face region extracted from track {track_idx}")
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (160x160 for InceptionResnetV1)
        face_resized = cv2.resize(face_rgb, (160, 160))
        
        # Convert to tensor and normalize to [-1, 1] range
        face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.unsqueeze(0).to(device)
        
        # Extract embedding using the model
        with torch.no_grad():
            embedding = embedding_model(face_tensor)
        
        return embedding[0].cpu().numpy()
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute similarity between two embeddings as 1 / (1 + distance).
        
        We use this transformation because:
        - Euclidean distance in embedding space: 0 means identical, larger means different
        - Similarity score: 1.0 means identical, 0.0 means very different
        
        This makes it more intuitive: higher score = more similar.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score between 0 and 1 (higher = more similar)
        """
        distance = euclidean(embedding1, embedding2)
        # Transform distance to similarity: closer distance = higher similarity
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def merge_tracks_by_similarity(self, embedding_model, similarity_threshold=0.7):
        """
        MAIN FUNCTION: Merge tracks based on similarity to anchor track.
        
        Algorithm:
        1. Find anchor track (track with most frames = target speaker)
        2. Extract embedding from anchor
        3. For each other track:
           - Extract embedding
           - Compute similarity to anchor
           - If similarity > threshold: mark for merging
           - Else: mark as separate (background person or noise)
        4. Merge all tracks marked for merging
        5. Return merged track with all frames sorted temporally
        
        Args:
            embedding_model: Pre-loaded InceptionResnetV1 model
            similarity_threshold: How similar a track needs to be to anchor
                                 to be considered the same person.
                                 Range: 0.0 to 1.0
                                 Typical: 0.6-0.8
                                 Lower threshold = more permissive (include more)
                                 Higher threshold = more strict (exclude more)
        
        Returns:
            Tuple of (merged_track, target_indices, summary_dict)
            - merged_track: Single track with all target speaker frames
            - target_indices: Which original tracks were merged
            - summary_dict: Statistics about the merging process
        """
        
        device = next(embedding_model.parameters()).device
        
        print("\n" + "="*70)
        print("SIMILARITY THRESHOLD TRACK MERGING")
        print("="*70)
        
        # Step 1: Find anchor track
        print(f"\nStep 1: Identify anchor track (track with most frames)")
        anchor_idx, anchor_frames = self.find_anchor_track()
        print(f"  Anchor track: Track {anchor_idx}")
        print(f"  Number of frames: {anchor_frames}")
        print(f"  (This is almost certainly your target speaker)")
        
        # Step 2: Extract anchor embedding
        print(f"\nStep 2: Extract anchor embedding from middle frame")
        try:
            anchor_embedding = self.extract_embedding(anchor_idx, embedding_model, device)
            print(f"  Successfully extracted embedding")
            print(f"  Embedding shape: {anchor_embedding.shape}")
        except Exception as e:
            print(f"  ERROR: Could not extract anchor embedding: {e}")
            raise
        
        # Step 3: Compare other tracks to anchor
        print(f"\nStep 3: Compare other tracks to anchor using similarity threshold")
        print(f"  Threshold: {similarity_threshold:.2f} (scale 0.0-1.0, higher = stricter)")
        print(f"\n  {'Track':<8} {'Frames':<10} {'Distance':<12} {'Similarity':<12} {'Decision':<12}")
        print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
        
        target_track_indices = [anchor_idx]  # Start with anchor
        comparison_results = []
        
        for track_idx in range(len(self.allTracks)):
            if track_idx == anchor_idx:
                # Skip anchor track itself
                num_frames = len(self.allTracks[track_idx]['frame'])
                print(f"  {track_idx:<8} {num_frames:<10} {'(anchor)':<12} {'(anchor)':<12} {'✓ ANCHOR':<12}")
                continue
            
            track = self.allTracks[track_idx]
            num_frames = len(track['frame'])
            
            try:
                # Extract embedding from this track
                other_embedding = self.extract_embedding(track_idx, embedding_model, device)
                
                # Compute similarity to anchor
                similarity = self.compute_similarity(anchor_embedding, other_embedding)
                
                # Get the raw distance for reporting
                distance = euclidean(anchor_embedding, other_embedding)
                
                # Decide if this track should be merged
                should_merge = similarity > similarity_threshold
                
                result = {
                    'track_idx': track_idx,
                    'num_frames': num_frames,
                    'distance': distance,
                    'similarity': similarity,
                    'should_merge': should_merge
                }
                comparison_results.append(result)
                
                # Print decision for this track
                decision = "✓ MERGE" if should_merge else "✗ IGNORE"
                print(f"  {track_idx:<8} {num_frames:<10} {distance:<12.3f} {similarity:<12.3f} {decision:<12}")
                
                if should_merge:
                    target_track_indices.append(track_idx)
            
            except Exception as e:
                print(f"  {track_idx:<8} {num_frames:<10} {'ERROR':<12} {'ERROR':<12} {'✗ ERROR':<12}")
                print(f"         Could not extract embedding: {e}")
                continue
        
        # Step 4: Merge selected tracks in temporal order
        print(f"\nStep 4: Merge selected tracks")
        print(f"  Selected tracks: {sorted(target_track_indices)}")
        
        # Collect all frames and bboxes from selected tracks
        all_frame_bbox_pairs = []
        
        for track_idx in target_track_indices:
            track = self.allTracks[track_idx]
            frames = track['frame']
            bboxes = track['bbox']
            
            # Create pairs of (frame_number, bbox) for sorting
            for frame_num, bbox in zip(frames, bboxes):
                all_frame_bbox_pairs.append({
                    'frame': frame_num,
                    'bbox': bbox,
                    'source_track': track_idx
                })
        
        # Sort by frame number to maintain temporal order
        all_frame_bbox_pairs.sort(key=lambda x: x['frame'])
        
        # Extract sorted frames and bboxes
        merged_frames = np.array([pair['frame'] for pair in all_frame_bbox_pairs])
        merged_bboxes = np.array([pair['bbox'] for pair in all_frame_bbox_pairs])
        
        # Create the merged track dictionary in the same format as track_shot output
        merged_track = {
            'frame': merged_frames,
            'bbox': merged_bboxes,
            'original_track_indices': target_track_indices,
            'num_original_tracks': len(target_track_indices),
            'anchor_track': anchor_idx,
            'similarity_threshold_used': similarity_threshold
        }
        
        # Step 5: Print summary
        print(f"\n  Total original tracks: {len(self.allTracks)}")
        print(f"  Tracks merged with target: {len(target_track_indices)}")
        print(f"  Total frames in merged track: {len(merged_frames)}")
        print(f"  Frame range: {merged_frames[0]} to {merged_frames[-1]}")
        
        # Create summary statistics
        summary = {
            'anchor_track_idx': anchor_idx,
            'anchor_num_frames': anchor_frames,
            'similarity_threshold': similarity_threshold,
            'target_track_indices': target_track_indices,
            'num_tracks_merged': len(target_track_indices),
            'total_frames_merged': len(merged_frames),
            'comparison_results': comparison_results,
            'num_tracks_rejected': len(self.allTracks) - len(target_track_indices)
        }
        
        print("="*70 + "\n")
        
        return merged_track, target_track_indices, summary


# # ============================================================================
# # INTEGRATION INTO YOUR EXISTING CODE
# # ============================================================================

# integration_example = """
# # In your main() function, after track_shot() produces allTracks:

# from merge_tracks_similarity import SimilarityThresholdTrackMerger
# import torch
# from facenet_pytorch import InceptionResnetV1

# # Initialize the embedding model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# embedding_model = InceptionResnetV1(pretrained='vggface2', classify=False)
# embedding_model.to(device)
# embedding_model.eval()

# # Create merger and merge tracks
# print("Merging tracks by similarity to anchor track...")
# start_time = time.time()

# merger = SimilarityThresholdTrackMerger(allTracks, video_args.pyframesPath)
# merged_track, target_indices, summary = merger.merge_tracks_by_similarity(
#     embedding_model,
#     similarity_threshold=0.65  # Adjust this if needed
# )

# end_time = time.time()
# print(f"Time taken to merge tracks: {end_time - start_time:.3f} seconds")

# # Replace allTracks with just the merged target speaker track
# allTracks = [merged_track]

# # Continue with face cropping and rest of pipeline
# print(f"\\nMerging complete. Selected {len(target_indices)} tracks containing target speaker.")
# print(f"Total frames for TSE: {len(merged_track['frame'])}")

# # Your existing code for face cropping continues unchanged:
# vidTracks = []
# for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
#     vidTracks.append(
#         crop_video(
#             video_args, track, os.path.join(video_args.pycropPath, "%05d" % ii)
#         )
#     )
# # ... rest of pipeline ...
# """

# print("SIMILARITY THRESHOLD TRACK MERGING MODULE")
# print("="*70)
# print(integration_example)
# print("="*70)