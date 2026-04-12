import sys, time, os, tqdm, torch, time, argparse, glob, subprocess, warnings, cv2, pickle
import numpy as np
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
import soundfile as sf
from batch_face import RetinaFace
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN
from facenet_pytorch import InceptionResnetV1

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from ..models.av_mossformer2_tse.faceDetector.s3fd import S3FD

from .decode import decode_one_audio_AV_MossFormer2_TSE_16K


def process_tse(args, model, device, data_reader, output_wave_dir):
    video_args = args_param()
    video_args.model = model
    video_args.device = device
    video_args.sampling_rate = args.sampling_rate
    args.device = device
    assert args.sampling_rate == 16000
    with torch.no_grad():
        for videoPath in data_reader:  # Loop over all video samples
            savFolder = videoPath.split(os.path.sep)[-1]
            video_args.savePath = f'{output_wave_dir}/{savFolder.split(".")[0]}/'
            video_args.videoPath = videoPath
            main(video_args, args)


def args_param():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nDataLoaderThread", type=int, default=12, help="Number of workers"
    )
    parser.add_argument(
        "--facedetScale",
        type=float,
        default=0.25,
        help="Scale factor for face detection, the frames will be scale to 0.25 orig",
    )
    parser.add_argument(
        "--minTrack", type=int, default=0, help="Number of min frames for each shot"
    )
    parser.add_argument(
        "--numFailedDet",
        type=int,
        default=10,
        help="Number of missed detections allowed before tracking is stopped",
    )
    parser.add_argument(
        "--minFaceSize", type=int, default=1, help="Minimum face size in pixels"
    )
    parser.add_argument(
        "--cropScale", type=float, default=0.40, help="Scale bounding box"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="The start time of the video"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="The duration of the video, when set as 0, will extract the whole video",
    )
    video_args = parser.parse_args()
    return video_args


# Main function
def main(video_args, args):
    # Initialization
    video_args.pyaviPath = os.path.join(video_args.savePath, "py_video")
    video_args.pyframesPath = os.path.join(video_args.savePath, "pyframes")
    video_args.pyworkPath = os.path.join(video_args.savePath, "pywork")
    video_args.pycropPath = os.path.join(video_args.savePath, "py_faceTracks")
    if os.path.exists(video_args.savePath):
        rmtree(video_args.savePath)
    os.makedirs(
        video_args.pyaviPath, exist_ok=True
    )  # The path for the input video, input audio, output video
    os.makedirs(video_args.pyframesPath, exist_ok=True)  # Save all the video frames
    os.makedirs(
        video_args.pyworkPath, exist_ok=True
    )  # Save the results in this process by the pckl method
    os.makedirs(
        video_args.pycropPath, exist_ok=True
    )  # Save the detected face clips (audio+video) in this process

    # Extract video
    video_args.videoFilePath = os.path.join(video_args.pyaviPath, "video.avi")
    # If duration did not set, extract the whole video, otherwise extract the video from 'video_args.start' to 'video_args.start + video_args.duration'

    start_time = time.time()
    if video_args.duration == 0:
        # 25 fps video is extracted from the original video
        command = (
            "ffmpeg -y -i %s -vf scale=720:-2:flags=fast_bilinear -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic"
            % (
                video_args.videoPath,
                video_args.nDataLoaderThread,
                video_args.videoFilePath,
            )
        )
    else:
        command = (
            "ffmpeg -y -i %s -vf scale=720:-2:flags=fast_bilinear -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic"
            % (
                video_args.videoPath,
                video_args.nDataLoaderThread,
                video_args.start,
                video_args.start + video_args.duration,
                video_args.videoFilePath,
            )
        )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Extract the video and save in %s \r\n" % (video_args.videoFilePath)
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken to extract video: {runtime:.3f} seconds")

    # Extract audio
    start_time = time.time()
    video_args.audioFilePath = os.path.join(video_args.pyaviPath, "audio.wav")
    command = (
        "ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic"
        % (
            video_args.videoFilePath,
            video_args.nDataLoaderThread,
            video_args.audioFilePath,
        )
    )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Extract the audio and save in %s \r\n" % (video_args.audioFilePath)
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken to extract audio: {runtime:.3f} seconds")

    # Extract the video frames
    start_time = time.time()
    command = "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % (
        video_args.videoFilePath,
        video_args.nDataLoaderThread,
        os.path.join(video_args.pyframesPath, "%06d.jpg"),
    )
    subprocess.call(command, shell=True, stdout=None)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Extract the frames and save in %s \r\n" % (video_args.pyframesPath)
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken to extract video frames: {runtime:.3f} seconds")

    # Scene detection for the video frames
    start_time = time.time()
    scene = scene_detect(video_args)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Scene detection and save in %s \r\n" % (video_args.pyworkPath)
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken for scene detection: {runtime:.3f} seconds")

    # Face detection for the video frames
    start_time = time.time()
    # faces = inference_video(video_args)
    faces = inference_video_retface(video_args)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Face detection and save in %s \r\n" % (video_args.pyworkPath)
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken for face detection: {runtime:.3f} seconds")

    # Face tracking
    start_time = time.time()
    allTracks, vidTracks = [], []
    for shot in scene:
        if (
            shot[1].frame_num - shot[0].frame_num >= video_args.minTrack
        ):  # Discard the shot frames less than minTrack frames
            allTracks.extend(
                track_shot(video_args, faces[shot[0].frame_num : shot[1].frame_num])
            )  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Face track and detected %d tracks \r\n" % len(allTracks)
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken for face tracking: {runtime:.3f} seconds")

    # Detect and keep only the target face track
    start_time = time.time()
    merged_target_track = merge_tracks_by_facial_identity(
        allTracks,
        video_args.pyframesPath,
        eps=0.5,                           # Distance threshold for clustering
        selection_method='center_distance', # Pick cluster closest to center
        sample_method='middle'              # Sample middle frame of each track
    )
    allTracks = [merged_target_track]
    print(f"Successfully merged tracks by facial identity")
    # Previous Method : 
    # target_face_idx = detect_target_face(allTracks, video_args.pyframesPath)
    # allTracks = [allTracks[target_face_idx]]
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken detect target face: {runtime:.3f} seconds")

    # Face clips cropping
    start_time = time.time()
    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(
            crop_video(
                video_args, track, os.path.join(video_args.pycropPath, "%05d" % ii)
            )
        )
    savePath = os.path.join(video_args.pyworkPath, "tracks.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(vidTracks, fil)
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Face Crop and saved in %s tracks \r\n" % video_args.pycropPath
    )
    fil = open(savePath, "rb")
    vidTracks = pickle.load(fil)
    fil.close()
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken to crop face clips: {runtime:.3f} seconds")

    # AVSE
    files = glob.glob("%s/*.avi" % video_args.pycropPath)
    files.sort()
    assert len(files) == 1
    fname = files[0].split("/")[-1].split(".")[-2]

    start_time = time.time()
    est_sources = evaluate_network(files, video_args, args)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Time taken for target speaker audio extraction: {runtime:.3f} seconds")

    # Save the estimated audio to wav format:
    est_audio = np.concatenate(est_sources, axis=0)
    max_value = np.max(np.abs(est_audio))
    if max_value > 1:
        est_audio /= max_value
    sf.write(video_args.pycropPath + f"/est_{fname}.wav", est_audio, 16000)

    rmtree(video_args.pyworkPath)
    rmtree(video_args.pyframesPath)


def split_to_chunks(in_path, out_path):

    command_avi = f"ffmpeg -i {in_path}.avi -c copy -map 0 -segment_time 10 -f segment {out_path}/%03d.avi"
    command_wav = f"ffmpeg -i {in_path}.wav -c copy -map 0 -segment_time 10 -f segment {out_path}/%03d.wav"

    output_avi = subprocess.call(command_avi, shell=True, stdout=None)
    output_wav = subprocess.call(command_wav, shell=True, stdout=None)
    print("Splitted the crop audio-video into 10s segments")


def detect_target_face(tracks, path):
    frame_h, frame_w = get_frame_hw(path)
    distances = []
    for track in tracks:
        dist = dist_centre(track, frame_h, frame_w)
        distances.append(dist)

    distances = np.array(distances)
    min_val_idx = np.argsort(distances)[0]
    return min_val_idx


def get_frame_hw(path):
    frame_paths = glob.glob(os.path.join(path, "*jpg"))
    frame = cv2.imread(frame_paths[0])
    h, w = frame.shape[:2]
    return h, w


def dist_centre(track, frame_h, frame_w):
    cx = (track["bbox"][:, 0] + track["bbox"][:, 2]) / 2
    cy = (track["bbox"][:, 1] + track["bbox"][:, 3]) / 2
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0
    dist = (cx - frame_cx) ** 2 + (cy - frame_cy) ** 2
    return dist.mean()


def scene_detect(video_args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([video_args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(video_args.pyworkPath, "scene.pckl")
    if sceneList == []:
        sceneList = [
            (videoManager.get_base_timecode(), videoManager.get_current_timecode())
        ]
    with open(savePath, "wb") as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write(
            "%s - scenes detected %d\n" % (video_args.videoFilePath, len(sceneList))
        )
    return sceneList


def load_img(p):
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def inference_video_retface(video_args):
    # GPU: Face detection, output is the list contains the face location and score in this frame

    gpu_id = 0 if torch.cuda.is_available() else -1
    DET = RetinaFace(gpu_id=gpu_id)
    flist = glob.glob(os.path.join(video_args.pyframesPath, "*.jpg"))
    flist.sort()

    start = time.time()
    with ThreadPoolExecutor(64) as ex:
        imgs = list(ex.map(load_img, flist))
    end = time.time()
    runtime = end - start
    print(f"Time taken to load all frames: {runtime:.3f} seconds")

    # if the image's max size is larger than 1080, it will be resized to 1080, -1 means no resize
    max_size = -1
    threshold = 0.8  # confidence threshold
    batch_size = 32  # images in a batch

    start = time.time()
    bboxes = DET(
        imgs,
        threshold=threshold,
        max_size=max_size,
        batch_size=batch_size,
        return_dict=True,
    )
    end = time.time()
    runtime = end - start
    print(f"Time taken to detect faces in all frames: {runtime:.3f} seconds")

    dets = []
    for fidx, fname in enumerate(flist):
        dets.append([])
        for bbox in bboxes[fidx]:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox["box"]).tolist(), "conf": bbox["score"]}
            )
        sys.stderr.write(
            "%s-%05d; %d dets\r" % (video_args.videoFilePath, fidx, len(dets[-1]))
        )

    savePath = os.path.join(video_args.pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(dets, fil)
    return dets


# Slower Alternative for Face Detection
def inference_video(video_args):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    DET = S3FD(device=video_args.device)
    flist = glob.glob(os.path.join(video_args.pyframesPath, "*.jpg"))
    flist.sort()
    dets = []
    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(
            imageNumpy, conf_th=0.9, scales=[video_args.facedetScale]
        )
        dets.append([])
        for bbox in bboxes:
            dets[-1].append(
                {"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]}
            )  # dets has the frames info, bbox info, conf info
        sys.stderr.write(
            "%s-%05d; %d dets\r" % (video_args.videoFilePath, fidx, len(dets[-1]))
        )
    savePath = os.path.join(video_args.pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(dets, fil)
    return dets


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(video_args, sceneFaces):
    # CPU: Face tracking
    iouThres = 0.0  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= video_args.numFailedDet:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > video_args.minTrack:
            frameNum = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])
            frameI = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if (
                max(
                    np.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                    np.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                )
                > video_args.minFaceSize
            ):
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks


def crop_video(video_args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(video_args.pyframesPath, "*.jpg"))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )  # Write video
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:  # Read the tracks
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = video_args.cropScale
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (
            video_args.audioFilePath,
            video_args.nDataLoaderThread,
            audioStart,
            audioEnd,
            audioTmp,
        )
    )
    output = subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = (
        "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic"
        % (cropFile, audioTmp, video_args.nDataLoaderThread, cropFile)
    )  # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")
    return {"track": track, "proc_track": dets}


def extract_representative_face_from_track(track, frames_path, sample_method='middle'):
    """
    Extract a single representative face image from a track for embedding computation.

    The track already has bboxes for all frames (interpolated), so we sample one
    frame and use its bbox to extract the face region.

    Args:
        track: Track dict with 'frame' array and 'bbox' array
        frames_path: Path to directory containing frame images
        sample_method: How to choose which frame to sample from the track.
                      Options: 'middle' (default), 'first', 'last'

    Returns:
        Tuple of (face_image, frame_number, bbox)
        - face_image: RGB numpy array of the extracted face
        - frame_number: Which frame this face was sampled from
        - bbox: The bounding box used to extract this face
    """

    frame_indices = track['frame']
    bboxes = track['bbox']

    # Choose which frame to sample from
    if sample_method == 'middle':
        # Middle frame is usually cleanest (not at track boundaries)
        sample_idx = len(frame_indices) // 2
    elif sample_method == 'first':
        sample_idx = 0
    elif sample_method == 'last':
        sample_idx = len(frame_indices) - 1
    else:
        raise ValueError(f"Unknown sample_method: {sample_method}")

    frame_num = int(frame_indices[sample_idx])
    bbox = bboxes[sample_idx]

    # Load the frame
    flist = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
    frame_path = flist[frame_num]
    frame = cv2.imread(frame_path)

    # Extract face region using the bbox
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Add small padding
    padding = 5
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)

    face_crop = frame[y1:y2, x1:x2]

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    return face_rgb, frame_num, bbox


def compute_embedding_for_face(face_image, model, device):
    """
    Compute a 512-dimensional embedding for a face image using InceptionResnetV1.

    Args:
        face_image: RGB face image as numpy array
        model: InceptionResnetV1 model instance
        device: torch device (cpu or cuda)

    Returns:
        512-dimensional embedding vector
    """

    # Resize to model input size
    face_resized = cv2.resize(face_image, (160, 160))

    # Convert to tensor
    face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0

    # Normalize
    face_tensor = (face_tensor - 0.5) / 0.5
    face_tensor = face_tensor.unsqueeze(0).to(device)

    # Extract embedding
    with torch.no_grad():
        embedding = model(face_tensor)

    return embedding[0].cpu().numpy()


def extract_embeddings_from_tracks(allTracks, frames_path, sample_method='middle'):
    """
    Extract one representative embedding from each track.

    This is much more efficient than computing embeddings for all detections.
    You only compute embeddings for one face per track (e.g., 5 embeddings
    instead of potentially hundreds).

    Args:
        allTracks: List of tracks from track_shot()
        frames_path: Path to frames directory
        sample_method: How to choose representative frame ('middle', 'first', 'last')

    Returns:
        Tuple of (embeddings_list, representative_info_list)
        - embeddings_list: List of 512-dim embedding vectors, one per track
        - representative_info_list: List of dicts with track_idx, frame_num, bbox
    """

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1(pretrained='vggface2', classify=False)
    model.to(device)
    model.eval()

    print(f"Extracting representative faces from {len(allTracks)} tracks...")
    print(f"Using device: {device}")

    embeddings = []
    representative_info = []

    for track_idx, track in enumerate(allTracks):
        try:
            # Extract face image from this track
            face_image, frame_num, bbox = extract_representative_face_from_track(
                track, frames_path, sample_method=sample_method
            )

            # Compute embedding
            embedding = compute_embedding_for_face(face_image, model, device)

            embeddings.append(embedding)
            representative_info.append({
                'track_idx': track_idx,
                'frame_num': frame_num,
                'bbox': bbox
            })

            print(f"  Track {track_idx}: sampled frame {frame_num}")

        except Exception as e:
            print(f"  Error processing track {track_idx}: {e}")
            continue

    print(f"Successfully extracted {len(embeddings)} embeddings\n")

    return embeddings, representative_info


def cluster_track_embeddings(embeddings, eps=0.5, min_samples=1):
    """
    Cluster track embeddings using DBSCAN to group tracks of the same person.

    Args:
        embeddings: List of embedding vectors
        eps: Distance threshold for DBSCAN (0.4-0.6 typical)
        min_samples: Minimum samples in a cluster (1 means each detection
                    counts as a potential cluster)

    Returns:
        Dict mapping cluster_id -> list of track indices in that cluster
    """

    embeddings_array = np.array(embeddings)

    print("Clustering track embeddings by facial identity...")

    # Fit DBSCAN
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(embeddings_array)

    # Group by cluster
    clusters = {}
    for track_idx, label in enumerate(labels):
        if label == -1:  # Skip noise
            print(f"  Track {track_idx} marked as noise/outlier")
            continue

        if label not in clusters:
            clusters[label] = []
        clusters[label].append(track_idx)

    print(f"Found {len(clusters)} identity clusters:\n")
    for cluster_id in sorted(clusters.keys()):
        track_indices = clusters[cluster_id]
        print(f"  Cluster {cluster_id}: tracks {track_indices} ({len(track_indices)} tracks total)")

    print()
    return clusters


def select_target_speaker_cluster(allTracks, clusters, frames_path, selection_method='center_distance'):
    """
    Select which cluster represents the target speaker.

    Args:
        allTracks: Original list of all tracks
        clusters: Dict from cluster_track_embeddings()
        frames_path: Path to frames directory (for computing frame dimensions)
        selection_method: How to choose the target cluster:
                         - 'center_distance': Pick cluster closest to frame center
                         - 'num_frames': Pick cluster with most total frames
                         - 'num_tracks': Pick cluster with most tracks

    Returns:
        Tuple of (target_cluster_id, cluster_stats)
        - target_cluster_id: Which cluster is the target speaker
        - cluster_stats: Dict with statistics about each cluster
    """

    # Get frame dimensions
    flist = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
    sample_frame = cv2.imread(flist[0])
    frame_h, frame_w = sample_frame.shape[:2]
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0

    print("="*70)
    print("CLUSTER STATISTICS FOR TARGET SPEAKER SELECTION")
    print("="*70)

    cluster_stats = {}

    for cluster_id, track_indices in clusters.items():
        # Compute statistics for this cluster
        num_tracks = len(track_indices)
        total_frames = sum(len(allTracks[tidx]['frame']) for tidx in track_indices)

        # Compute average distance from center
        total_dist = 0.0
        total_bbox_count = 0
        for tidx in track_indices:
            track = allTracks[tidx]
            bboxes = track['bbox']
            for bbox in bboxes:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                dist = np.sqrt((cx - frame_cx)**2 + (cy - frame_cy)**2)
                total_dist += dist
                total_bbox_count += 1

        avg_dist_from_center = total_dist / total_bbox_count if total_bbox_count > 0 else float('inf')

        cluster_stats[cluster_id] = {
            'num_tracks': num_tracks,
            'total_frames': total_frames,
            'avg_dist_from_center': avg_dist_from_center,
            'track_indices': track_indices
        }

        print(f"\nCluster {cluster_id}:")
        print(f"  Tracks: {track_indices}")
        print(f"  Number of tracks: {num_tracks}")
        print(f"  Total frames: {total_frames}")
        print(f"  Avg distance from center: {avg_dist_from_center:.1f} pixels")

    # Select target cluster
    print("\n" + "="*70)

    if selection_method == 'center_distance':
        target_cluster_id = min(
            cluster_stats.keys(),
            key=lambda cid: cluster_stats[cid]['avg_dist_from_center']
        )
        reason = f"closest to center ({cluster_stats[target_cluster_id]['avg_dist_from_center']:.1f} pixels)"

    elif selection_method == 'num_frames':
        target_cluster_id = max(
            cluster_stats.keys(),
            key=lambda cid: cluster_stats[cid]['total_frames']
        )
        reason = f"most frames ({cluster_stats[target_cluster_id]['total_frames']} frames)"

    elif selection_method == 'num_tracks':
        target_cluster_id = max(
            cluster_stats.keys(),
            key=lambda cid: cluster_stats[cid]['num_tracks']
        )
        reason = f"most tracks ({cluster_stats[target_cluster_id]['num_tracks']} tracks)"

    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")

    print(f"TARGET SPEAKER: Cluster {target_cluster_id}")
    print(f"Reason: {reason}")
    print("="*70 + "\n")

    return target_cluster_id, cluster_stats


def merge_tracks_of_same_person(allTracks, target_track_indices):
    """
    Merge multiple tracks that belong to the same person into a single unified track.

    This is the crucial part: we need to respect the temporal order of frames
    and properly combine the interpolated bounding boxes.

    Args:
        allTracks: Original list of all tracks
        target_track_indices: List of track indices to merge (e.g., [0, 2, 3])

    Returns:
        Single merged track dict with 'frame' and 'bbox' arrays
    """

    print(f"Merging tracks {target_track_indices}...")

    # Collect all frame-bbox pairs from the tracks we're merging
    frame_bbox_pairs = []

    for track_idx in target_track_indices:
        track = allTracks[track_idx]
        frames = track['frame']
        bboxes = track['bbox']

        # Each frame has a corresponding bbox
        for frame_num, bbox in zip(frames, bboxes):
            frame_bbox_pairs.append({
                'frame': frame_num,
                'bbox': bbox,
                'original_track': track_idx
            })

    # Sort by frame number to get temporal order
    frame_bbox_pairs.sort(key=lambda x: x['frame'])

    # Extract sorted frames and bboxes
    merged_frames = np.array([pair['frame'] for pair in frame_bbox_pairs])
    merged_bboxes = np.array([pair['bbox'] for pair in frame_bbox_pairs])

    merged_track = {
        'frame': merged_frames,
        'bbox': merged_bboxes,
        'original_track_indices': target_track_indices,
        'num_original_tracks': len(target_track_indices)
    }

    print(f"Merged track contains {len(merged_frames)} frames")
    print(f"Frame range: {merged_frames[0]} to {merged_frames[-1]}")
    print(f"Original separate frames from {len(target_track_indices)} tracks combined\n")

    return merged_track


def merge_tracks_by_facial_identity(allTracks, frames_path, eps=0.5,
                                     selection_method='center_distance',
                                     sample_method='middle'):
    """
    MAIN FUNCTION: Cluster tracks by facial identity and merge target speaker tracks.

    This function orchestrates the entire process:
    1. Extract one representative face from each track
    2. Compute embeddings for facial identity
    3. Cluster embeddings to group same-person tracks
    4. Select which cluster is the target speaker
    5. Merge all target speaker tracks into one unified track

    Args:
        allTracks: List of tracks from track_shot()
        frames_path: Path to extracted video frames
        eps: DBSCAN epsilon parameter (0.4-0.6 typical)
        selection_method: How to pick target speaker ('center_distance', 'num_frames', 'num_tracks')
        sample_method: Which frame to sample from each track ('middle', 'first', 'last')

    Returns:
        Single merged track containing all target speaker detections
    """

    print("\n" + "="*70)
    print("MERGING TRACKS BY FACIAL IDENTITY")
    print("="*70 + "\n")

    # Step 1: Extract embeddings from each track
    embeddings, representative_info = extract_embeddings_from_tracks(
        allTracks, frames_path, sample_method=sample_method
    )

    if len(embeddings) == 0:
        raise ValueError("No embeddings could be extracted from tracks")

    # Step 2: Cluster embeddings
    clusters = cluster_track_embeddings(embeddings, eps=eps, min_samples=1)

    if len(clusters) == 0:
        raise ValueError("No clusters formed. Try increasing eps parameter.")

    # Step 3: Select target speaker cluster
    target_cluster_id, cluster_stats = select_target_speaker_cluster(
        allTracks, clusters, frames_path, selection_method=selection_method
    )

    # Step 4: Get the track indices for target cluster
    target_track_indices = cluster_stats[target_cluster_id]['track_indices']

    # Step 5: Merge the target tracks
    merged_track = merge_tracks_of_same_person(allTracks, target_track_indices)

    return merged_track


# ============================================================================
# Example usage
# ============================================================================

# if __name__ == "__main__":
#     """
#     Example of how to use this module in your preprocessing pipeline.
#     """

#     # In your main() function, after you've obtained allTracks from track_shot():

#     # merged_target_track = merge_tracks_by_facial_identity(
#     #     allTracks,
#     #     video_args.pyframesPath,
#     #     eps=0.5,
#     #     selection_method='center_distance',
#     #     sample_method='middle'
#     # )
#     #
#     # # Replace allTracks with the merged track
#     # allTracks = [merged_target_track]
#     #
#     # # Now continue with the rest of your pipeline (face cropping, etc.)

#     print("Module ready for import. See docstrings for usage.")


def evaluate_network(files, video_args, args):

    est_sources = []
    for file in tqdm.tqdm(files, total=len(files)):

        fileName = os.path.splitext(file.split(os.path.sep)[-1])[
            0
        ]  # Load audio and video
        audio, _ = sf.read(
            os.path.join(video_args.pycropPath, fileName + ".wav"), dtype="float32"
        )

        video = cv2.VideoCapture(os.path.join(video_args.pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                videoFeature.append(face)
            else:
                break

        video.release()
        visual = np.array(videoFeature) / 255.0
        visual = (visual - 0.4161) / 0.1688

        length = int(audio.shape[0] / 16000 * 25)
        if visual.shape[0] < length:
            visual = np.pad(
                visual,
                ((0, int(length - visual.shape[0])), (0, 0), (0, 0)),
                mode="edge",
            )

        audio /= np.max(np.abs(audio))
        audio = np.expand_dims(audio, axis=0)
        visual = np.expand_dims(visual, axis=0)

        inputs = (audio, visual)
        est_source = decode_one_audio_AV_MossFormer2_TSE_16K(
            video_args.model, inputs, args
        )

        est_sources.append(est_source)

    return est_sources
