import argparse
from clearvoice import ClearVoice


def main(args):
    myClearVoice = ClearVoice(
        task="target_speaker_extraction", model_names=["AV_MossFormer2_TSE_16K"]
    )

    myClearVoice(
        input_path=args.input_path, online_write=True, output_path=args.output_path
    )

    print("process completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ClearVoice target speaker extraction"
    )

    # inp_path = "/content/ClearerVoice-Studio/clearvoice/samples/path_to_input_videos_tse/*.mp4"
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input video file or directory",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        default="/content/ClearerVoice-Studio/clearvoice/samples/path_to_output_videos_tse",
        help="Path to output directory",
    )

    args = parser.parse_args()
    main(args)
