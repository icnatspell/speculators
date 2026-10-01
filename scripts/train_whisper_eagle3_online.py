"""Train native EAGLE-3 on Whisper features using the shared online pipeline."""

from train_whisper_online import main

if __name__ == "__main__":
    main(default_algorithm="eagle3")
