"""Report the TensorFlow runtime used by the Ubuntu training environment."""

import tensorflow as tf


def main() -> None:
    print(f"TensorFlow: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Visible GPUs: {len(gpus)}")
    for index, gpu in enumerate(gpus):
        print(f"  GPU {index}: {gpu.name}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("    memory growth: enabled")
        except RuntimeError as error:
            print(f"    memory growth: unchanged ({error})")


if __name__ == "__main__":
    main()
