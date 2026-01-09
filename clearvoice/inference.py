from clearvoice import ClearVoice

def main():
    myClearVoice = ClearVoice(
        task="target_speaker_extraction", model_names=["AV_MossFormer2_TSE_16K"]
    )

    myClearVoice(
        input_path="/content/ClearerVoice-Studio/clearvoice/samples/path_to_input_videos_tse/bj06_split_00.mp4", 
        online_write=True, 
        output_path="/content/ClearerVoice-Studio/clearvoice/samples/path_to_output_videos_tse/"
        )
    

    print("process completed")


if __name__ == "__main__":
    main()
