from clearvoice import ClearVoice

def main():
    myClearVoice = ClearVoice(
        task="target_speaker_extraction", model_names=["AV_MossFormer2_TSE_16K"]
    )

    myClearVoice(
        input_path="/content/ClearerVoice-Studio/clearvoice/samples/scp/video_samples.scp", 
        online_write=True, 
        output_path="/content/drive/MyDrive/game-recordings/AV-BJ/clean_moss_2/",
        # output_path="/content/ClearerVoice-Studio/clearvoice/samples/path_to_output_videos_tse/"
        )
    

    print("process completed")


if __name__ == "__main__":
    main()
